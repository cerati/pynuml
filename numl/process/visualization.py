import pandas as pd, torch, torch_geometric as tg
import numpy as np
import sys
sys.path.append("..")
from labels import *
from labels import standard
from graph import *
from core.file import NuMLFile
import plotly.express as px
from collections import deque
    
def single_plane_graph_vis(evt, l=standard):
    """Process an event into graphs"""

    # get energy depositions, find max contributing particle, and ignore any hits with no truth
    evt_edep = evt["edep_table"]
    evt_edep = evt_edep.loc[evt_edep.groupby("hit_id")["energy_fraction"].idxmax()]
    evt_hit = evt_edep.merge(evt["hit_table"], on="hit_id", how="inner")

    # skip events with fewer than 50 simulated hits in any plane
    for i in range(3):
        if (evt_hit.global_plane==i).sum() < 50: return

    # get labels for each particle
    evt_part = evt["particle_table"]
    evt_part = l.semantic_label(evt_part)

    parent_dict = {0 : ["No Info", "No Info"]}
    for g,t,m in zip(evt_part['g4_id'], evt_part['type'], evt_part['momentum']):
        if g not in parent_dict:
            parent_dict[g] = [t, m]

    # join the dataframes to transform particle labels into hit labels
    evt_hit = evt_hit.merge(evt_part, on="g4_id", how="inner")

    planes = []
    for p, plane in evt_hit.groupby("local_plane"):

        # Reset indices
        plane = plane.reset_index(drop=True).reset_index()

        plane['parent_type'] = plane.apply(lambda row: parent_dict[row['parent_id']][0], axis=1)
        plane['parent_momentum'] = plane.apply(lambda row: parent_dict[row['parent_id']][1], axis=1)

        planes.append(plane)

    return planes

def extract_hierarchy(planes_arr, evt_part):
    # assuming tree structure
    def bfs(adj_list, info):
        rows = []
        indices = []
        q = deque()
        q.append(0)
        while len(q):
            size = len(q)
            for _ in range(size):
                node = q.popleft()
                if node:
                    rows.append([info[node][0], info[node][1], info[node][2], adj_list[node]])
                else:
                    rows.append([None, None, 0, adj_list[node]])
                indices.append(node)
                q.extend(adj_list[node])
        
        hierarchy = pd.DataFrame(rows, columns=['type', 'momentum', 'hit_count', 'neighbors'], index=indices)
        return hierarchy

              
    h_planes = []
    for plane in planes_arr:
        adj_list = {}
        for p in plane['parent_id'].unique():
            adj_list[p] = plane[plane['parent_id'] == p]['g4_id'].unique()
        for p in plane['g4_id'].unique():
            if p not in adj_list:
                adj_list[p] = []
        
        info = {p : (evt_part[evt_part.g4_id == p]['type'].values[0], 
                     evt_part[evt_part.g4_id == p]['momentum'].values[0], 
                     plane[plane.g4_id == p].shape[0]) for p in plane['g4_id'].unique()}  
        h_planes.append(bfs(adj_list, info))
        
    return h_planes

# def extract_hierarchy(part, key):
#     # assuming tree structure
#     def bfs(adj_list, info):
#         rows = []
#         indices = []
#         q = deque()
#         q.append(0)
#         while len(q):
#             size = len(q)
#             for _ in range(size):
#                 node = q.popleft()
#                 if node:
#                     rows.append([info[node][0], info[node][1], adj_list[node]])
#                 else:
#                     rows.append([None, None, adj_list[node]])
#                 indices.append(node)
#                 q.extend(adj_list[node])
        
#         hierarchy = pd.DataFrame(rows, columns=['type', 'momentum', 'neighbors'], index=indices)
#         return hierarchy

#     evt_part = part.loc[key].reset_index(drop=True)
   
#     adj_list = {}
#     for p in evt_part['parent_id'].unique():
#         adj_list[p] = evt_part[evt_part['parent_id'] == p]['g4_id'].tolist()
#     for p in evt_part['g4_id']:
#         if p not in adj_list:
#             adj_list[p] = []

#     evt_part = part.loc[key].reset_index(drop=True)
#     info = {p : (evt_part[evt_part.g4_id == p]['type'].values[0], 
#                  evt_part[evt_part.g4_id == p]['momentum'].values[0]) for p in evt_part['g4_id']}  
        
#     return bfs(adj_list, info)

def handle_planes(planes_arr):
    particle_dtype = pd.CategoricalDtype(["pion",
      "muon",
      "kaon",
      "hadron",
      "shower",
      "michel",
      "delta",
      "diffuse",
      "invisible"], ordered = True)

    for i in range(3):
        planes_arr[i].semantic_label = pd.Categorical(planes_arr[i].semantic_label).from_codes(codes = planes_arr[i].semantic_label, dtype = particle_dtype)
        planes_arr[i].start_process = planes_arr[i].start_process.map(lambda s : s.decode('utf-8'))
        planes_arr[i].end_process = planes_arr[i].end_process.map(lambda s : s.decode('utf-8'))

    return pd.concat(planes_arr)


def plot_event(df, print_out=True, write=None):
    color_dict = {"pion" : "yellow",
      "muon" : "green",
      "kaon" : "black",
      "hadron" : "blue",
      "shower" : "red",
      "michel" : "purple",
      "delta" : "pink",
      "diffuse" : "orange",
      "invisible" : "white"}
    
    fig = px.scatter(df, x="local_wire", y="local_time", color="semantic_label", color_discrete_map=color_dict, facet_col="local_plane",
                     hover_data=["g4_id","type", "momentum", "start_process", "end_process", "parent_id", "parent_type", "parent_momentum"])

    fig.update_layout(
        width = 1200,
        height = 500,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="LightSteelBlue",
    )
    if print_out:
        fig.show()

    if write:
        fig.write_html(write)

def concat_events(fname):
    f = NuMLFile(fname)
    evt = f.get_dataframe("event_table", ["event_id"])
    hit = f.get_dataframe("hit_table")
    part = f.get_dataframe("particle_table", ["event_id", "g4_id", "parent_id", "type", "momentum", "start_process", "end_process"])
    edep = f.get_dataframe("edep_table")
    sp = f.get_dataframe("spacepoint_table")
    
    data = pd.DataFrame()
    
    for i,key in enumerate(evt.index):
        planes = single_plane_graph_vis(key, hit, part, edep, sp)
        if planes:
            planes = handle_planes(planes)
            data = pd.concat([data, planes], ignore_index=True)
            
    return data

def label_counts(data):
    counts = data['semantic_label'].value_counts()
    ax = counts.plot(kind='bar')
    ax.set_ylabel("count")

def histogram_slice(data, metric, log_scale=False, write=False):
    color_dict = {"pion" : "yellow",
      "muon" : "green",
      "kaon" : "black",
      "hadron" : "blue",
      "shower" : "red",
      "michel" : "purple",
      "delta" : "pink",
      "diffuse" : "orange",
      "invisible" : "white"}

    fig = px.histogram(data, x=metric, color="semantic_label", color_discrete_map=color_dict, facet_col="local_plane", log_y=log_scale)

    fig.update_layout(
        width = 1200,
        height = 500,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="LightSteelBlue",
    )
    fig.show()
    
    if write: fig.write_html("hist_nue.html")
