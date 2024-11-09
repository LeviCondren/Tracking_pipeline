import os
import sys
import numpy as np
from numpy.linalg import norm
import pandas as pd
import matplotlib.pyplot as plt

ATLASradii = np.linspace(3.10,53.0,25)
ATLASlength = 320
print("running script")

train_size = 100
validate_size = 0
test_size = 100
fourierDimTrain = 1
fourierDimTest = 1
data_combination_train = "SM and Signal"
data_combination_test = "SM and Signal"
#new_output_folder = "dim_1_dim_1_sm_and_signal_train_sm_and_signal_test"
new_output_folder = "test"

signal_PID = 15
bField = 1
max_B_radius = np.max(ATLASradii)

input_dir = 'Event_files_93000'
output_dir = 'combined_hit_particle_files'

signal_tracks_per_event = 1
fourierRadiiTrain = [10 for n in range(2 * fourierDimTrain)]
fourierRadiiTest = [10 for n in range(2 * fourierDimTest)]
fourierCenters = np.zeros(2 * fourierDimTrain)
Lambda = np.max(ATLASradii)
min_dist_to_detector_layer = 0.001
times = np.linspace(0,100,10000000)

plotting = False
plotting_datatype = 'train'
num_plotted_samples = 5
plotting_save_file = 'plot_4_of_tracks'
plot_title = "Tracks Generated in Schwartz Space"

chunk_size = 40

#if we wanted to make a different volume in fourier space, we could easily make a new function, ie. sample_from_cube or something analogous 
def sample_from_ball(chunk_size, radius, center = np.zeros(3)):
    points = []
    for i in range(chunk_size):
        vec = np.random.uniform(radius/2,-radius/2, 3)
        if norm(vec) > radius:
            i -= 1
            pass
        
        else:
            vec += center
            points.append(vec)
    return np.array(points)

def make_tracks_from_fourier_balls(chunk_size, fourierDim, radii, centers):
    hyper_fourier_points = []
    loop_order = sorted(range(-fourierDim, fourierDim + 1), key=lambda x: -abs(x))[:-1]
    for dimension in loop_order:
        hyper_fourier_points.append(sample_from_ball(chunk_size,radius = radii[dimension],center=centers[dimension]))
    return np.array(hyper_fourier_points)

def layerIDfunction(r):
    difference_array = np.absolute(ATLASradii - r)
    index = difference_array.argmin() + 1
    return index

def fourierExpand(fourierDim, Lambda, t):
    fourList = []
    loop_order = sorted(range(-fourierDim, fourierDim + 1), key=lambda x: -abs(x))[:-1]
    for dimension in loop_order:
        fourList.append(np.cos(2 * np.pi * dimension * t/Lambda) - 1)
    return np.array(fourList).T

def tracks_cylindrical_fourier_balls(t,fourierDim, Lambda, chunk_size, radii, centers):
    cartesian_curve = fourierExpand(fourierDim, Lambda, t)[:,np.newaxis,np.newaxis,:] * np.transpose(make_tracks_from_fourier_balls(chunk_size,fourierDim, 
                                                                                                                                    radii, centers), axes = [2,1,0])
    cartesian_curve = np.sum(cartesian_curve,axis=-1)
    x = cartesian_curve[:,0,:]
    y = cartesian_curve[:,1,:]
    z = cartesian_curve[:,2,:]
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y,x)
    return np.array(np.transpose(np.array([r,phi,z]),axes = [1,2,0]))

def map_curve_to_hits(curve, min_dist_to_detector_layer):

    #delete points not close enough to any detector layer
    close_points_index = np.argwhere((np.abs((curve[:,0])[:, np.newaxis]-ATLASradii[np.newaxis, :])).min(axis = 1) < min_dist_to_detector_layer).flatten()
    curve = curve[close_points_index, :]

    #group points based on which layer they are closest to
    closest_layer_per_point = np.argmin(np.abs((curve[:,0])[np.newaxis,:] - ATLASradii[:,np.newaxis]), axis = 0)

    #make lists of -1 and 1 to represent when the curve crosses a layer
    sign_list = np.sign(curve[:,0] - ATLASradii[closest_layer_per_point])

    #multiply by index of appropriate layer. We'll get list like (-1, -1, -1, 1, 1, 1 , -2, -2, 2, 2, -3 , -3, ...)
    sign_list = sign_list * (closest_layer_per_point + np.ones_like(closest_layer_per_point))

    hit_indices = np.argwhere(np.abs(np.roll(sign_list, 1) + sign_list) < 1/2)
    hit_indices = hit_indices[(hit_indices > 0) & (hit_indices < len(curve))]
    hits = curve[hit_indices, :]

    #append the layer id to the hits
    layerID = ((closest_layer_per_point + np.ones_like(closest_layer_per_point))[hit_indices])
    hits = np.concatenate((hits, (layerID[np.newaxis, :]).T), axis = 1)
    return hits

def make_list_of_hits_from_fourier_balls(chunk_size, Radii, fourierDim,times ,Centers ,Lambda = np.max(ATLASradii), min_dist_to_detector_layer = 0.001):
    #Tracks has shape (chunk_size_train,fourDimTrain)

    Tracks = tracks_cylindrical_fourier_balls(times,fourierDim,Lambda,chunk_size,Radii, Centers)

    signal_hits = [] 
    for track in range(chunk_size):

        #curve has shape (time steps, coordinates)
        curve = Tracks[:,track,:]

        intersection_points = map_curve_to_hits(curve, min_dist_to_detector_layer)
        intersection_points = np.hstack((((track + 1) * np.ones(len(intersection_points))[np.newaxis, :]).T, intersection_points))

        intersection_points_df = pd.DataFrame(intersection_points, columns = ['particle_id','r', 'phi','z', 'layer_id'])

        #Delete everything past when the particle leaves the detector

        intersection_points_df = intersection_points_df.loc[:intersection_points_df[(intersection_points_df['r'] >= max_B_radius)].index.min()]
        intersection_points_df = intersection_points_df.loc[:intersection_points_df[np.abs(intersection_points_df['z']) >= ATLASlength/2].index.min()]
        signal_hits.append(intersection_points_df)
    return signal_hits

def make_track_plot(tracks, num_plotted_samples):
    # TracksTrain has shape (chunk_size_train, fourDimTrain)
    
    # Creating figure
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")

    # Plot all tracks
    for track in range(num_plotted_samples):
        # curve has shape (time steps, coordinates)
        curve = tracks[track]
        r = curve['r']
        phi = curve['phi']
        z = curve['z']

        x = r * np.cos(phi)
        y = r * np.sin(phi)

        # Add each track to the same 3D plot
        ax.scatter3D(x, y, z, s=3, label=f"Track {track}")

    # Create concentric cylinders
    theta = np.linspace(0, 2 * np.pi, 100)  # Angle for the circular base of the cylinder
    z_range = np.linspace(np.min([np.min(df['z']) for df in tracks]), np.max([np.max(df['z']) for df in tracks]), 100)  # z-axis range

    for radius in ATLASradii[22:]:
        # Meshgrid for the cylinder
        Theta, Z = np.meshgrid(theta, z_range)
        X = radius * np.cos(Theta)
        Y = radius * np.sin(Theta)
        
        # Plot each cylinder with semi-transparency
        ax.plot_surface(X, Y, Z, color='cyan', alpha=0.2, rstride=5, cstride=5)

    # Final plot settings
    plt.title(plot_title)
    plt.legend()
    plt.savefig(plotting_save_file)

def make_bkg_hits(input_dir):
    bkg_hits = []
    files = os.listdir(input_dir)

    for event in files:
        file_path = os.path.join(input_dir, event)
        particles_df = pd.read_csv(file_path)
        
        particles_df['helical_radius'] = 100 * np.sqrt(particles_df['px']**2 + particles_df['py']**2)/bField
        particles_df.drop(particles_df[particles_df['helical_radius'] < ATLASradii[0]/2].index, inplace = True)
        particles_df['precession_frequency'] = bField/np.sqrt(
            particles_df['px']**2 + particles_df['py']**2 + particles_df['pz']**2 + particles_df['Mass']**2)
        particles_df['vz'] = 100 * particles_df['vz']

        particles_df['argArcCos'] = particles_df.apply(lambda row:np.array(
            [(1 - (1/2) * (ATLASradii[r]/(row['helical_radius'])) **2).astype(float) for r in range(len(ATLASradii))]), axis = 1)
        
        particles_df = particles_df.explode('argArcCos').reset_index(drop = True)
        particles_df = particles_df[particles_df['argArcCos'].apply(lambda x: np.all(np.abs(x) <= 1))]
        particles_df['argArcCos'] = particles_df['argArcCos'].astype(float)
        # find intersection_times here
        particles_df['intersection_times'] = ((1/particles_df['precession_frequency']) * np.arccos(particles_df['argArcCos'])).astype(float)

        hits_df = pd.DataFrame()

        hits_df['particle_id'] = particles_df['particle_id']
        hits_df['r'] = particles_df['helical_radius'] * np.sqrt(2 - 2 * np.cos(particles_df['precession_frequency'] * particles_df['intersection_times']))
        hits_df['phi'] = np.arctan2(particles_df['helical_radius'] * np.sin(particles_df['precession_frequency'] * particles_df['intersection_times']),
                                    particles_df['helical_radius'] * (np.cos(particles_df['precession_frequency'] * particles_df['intersection_times']) - 1))
        hits_df['z'] = particles_df['vz'] * particles_df['intersection_times']
        hits_df['layer_id'] = hits_df.apply(lambda row: layerIDfunction(row['r']), axis = 1)
        hits_df['hit_id'] = hits_df.index + 1

        bkg_hits.append(hits_df)
    
    return bkg_hits

def prepare_signal_dfs(chunk, chunk_size, fourierRadii, fourierDim, times, fourierCenters, Lambda, min_dist_to_detector_layer, 
                       event_id_minus_event, final_iteration = False, signal_hits = None, remaining_events_after_chunks = None):
    if final_iteration == False:
        signal_hits = make_list_of_hits_from_fourier_balls(chunk_size, fourierRadii, fourierDim,times , fourierCenters,Lambda , 
                                                           min_dist_to_detector_layer)
        range_for_events = chunk_size
    elif final_iteration == True:
        range_for_events = remaining_events_after_chunks
    if (num_plotted_samples > chunk_size) & (plotting == True):
        raise ValueError("num_plotted_samples must be smaller than chunk_size")
    if (plotting == True) & (chunk == 0):
        make_track_plot(signal_hits, num_plotted_samples)

    for item in range(range_for_events):
        if final_iteration == False:
            event_id = event_id_minus_event + item
        elif final_iteration == True:
            event_id = event_id_minus_event
        signal_hits_df = signal_hits[item]

        signal_particle_df = pd.DataFrame({
            'particle_id':[1],
            'vx':[10],
            'vy':[10],
            'vz':[10],
            'px':[10],
            'py':[10],
            'pz':[10],
            'charge':[1],
            'PID':[signal_PID],
            'Event#':[event_id],
            'Mass':[1000]
        })
        signal_hits_df['particle_id'] = 1

        signal_particle_df.to_csv(os.path.join(output_dir,new_output_folder,f'event{event_id + 100000000}-particles.csv'), index=False) 
        signal_hits_df.to_csv(os.path.join(output_dir,new_output_folder,f'event{event_id + 100000000}-hits.csv'), index=False)
        del signal_particle_df
        del signal_hits_df
        
def prepare_SM_dfs(chunk, chunk_size, event_id_minus_event, bkg_hits, final_iteration = False, remaining_events_after_chunks = None):
    if final_iteration == False:
        range_for_events = chunk_size
    elif final_iteration == True:
        range_for_events = remaining_events_after_chunks

    for item in range(range_for_events):
        if final_iteration == False:
            event_id = event_id_minus_event + item
        elif final_iteration == True:
            event_id = event_id_minus_event
        
        bkg_hits_df = bkg_hits[event_id - 1]
        bkg_hits_df.to_csv(os.path.join(output_dir,new_output_folder,f'event{event_id + 100000000}-hits.csv'), index=False)
        del bkg_hits_df

        bkg_particles_df = pd.read_csv(os.path.join(input_dir, f'Event_{event_id-1}_bkg.csv'))
        bkg_particles_df['Event#'] = event_id
        bkg_particles_df.to_csv(os.path.join(output_dir,new_output_folder,f'event{event_id + 100000000}-particles.csv'), index=False)
        del bkg_particles_df
        del event_id


def combine_SM_and_signal_dfs(chunk, chunk_size, fourierRadii, fourierDim, times, fourierCenters, Lambda, min_dist_to_detector_layer, 
                              event_id_minus_event, bkg_hits, final_iteration = False, signal_hits = None, remaining_events_after_chunks = None):
    if final_iteration == False:
        signal_hits = make_list_of_hits_from_fourier_balls(chunk_size, fourierRadii, fourierDim,times , fourierCenters,Lambda, 
                                                           min_dist_to_detector_layer)
        range_for_events = chunk_size
        print("made sig hits")
    elif final_iteration == True:
        range_for_events = remaining_events_after_chunks
    if num_plotted_samples > chunk_size:
        raise ValueError("num_plotted_samples must be smaller than chunk_size")
    if (plotting == True) & (chunk == 0):
        make_track_plot(signal_hits, num_plotted_samples)

    for item in range(range_for_events):
        if final_iteration == False:
            event_id = event_id_minus_event + item
        elif final_iteration == True:
            event_id = event_id_minus_event
        signal_hits_df = signal_hits[item]
        print("processed an event")
        bkg_hits_df = bkg_hits[event_id - 1]
        bkg_particles_df = pd.read_csv(os.path.join(input_dir, f'Event_{event_id-1}_bkg.csv'))

        #make random signal particle_id
        existing_ids = set(bkg_particles_df['particle_id'])
        new_particle_id = np.random.randint(1, max(existing_ids) + 1)
        while new_particle_id in existing_ids:
            new_particle_id = np.random.randint(1, max(existing_ids) + 1)

        signal_particle_df = pd.DataFrame({
            'particle_id':[new_particle_id],
            'vx':[10],
            'vy':[10],
            'vz':[10],
            'px':[10],
            'py':[10],
            'pz':[10],
            'charge':[1],
            'PID':[signal_PID],
            'Event#':[event_id],
            'Mass':[1000]
        })
        signal_hits_df['particle_id'] = new_particle_id
        bkg_particles_df['Event#'] = event_id

        #insert signal in particles df in random position
        combined_particles_df = pd.concat([bkg_particles_df, signal_particle_df], ignore_index=True)
        combined_particles_df = combined_particles_df.sample(frac=1).reset_index(drop=True)
        combined_particles_df = combined_particles_df.sort_values(by='particle_id').reset_index(drop=True)
        combined_particles_df['particle_id'] = combined_particles_df['particle_id'].astype(int)
        combined_particles_df['PID'] = combined_particles_df['PID'].astype(int)
        combined_particles_df.to_csv(os.path.join(output_dir,new_output_folder,f'event{event_id + 100000000}-particles.csv'), index=False) 
        del combined_particles_df
        del bkg_particles_df
        del signal_particle_df

        #insert signal in hits df
        gap_size = signal_hits_df.shape[0]
        insertion_index = np.searchsorted(bkg_hits_df['particle_id'], new_particle_id)
        empty_rows = pd.DataFrame(np.nan, index=range(gap_size), columns=bkg_hits_df.columns)
        combined_hits_df = pd.concat([bkg_hits_df.iloc[:insertion_index], empty_rows, bkg_hits_df.iloc[insertion_index:]]).reset_index(drop=True)

        signal_hits_df['hit_id'] = 1
        combined_hits_df.iloc[insertion_index:insertion_index+gap_size] = signal_hits_df.values
        combined_hits_df['hit_id'] = combined_hits_df.index + 1
        combined_hits_df['particle_id'] = combined_hits_df['particle_id'].astype(int)
        combined_hits_df.to_csv(os.path.join(output_dir,new_output_folder,f'event{event_id + 100000000}-hits.csv'), index=False)
        del combined_hits_df
    

def make_files(input_dir, datatype, signal_tracks_per_event, fourierRadii,fourierDim ,times, fourierCenters ,Lambda = np.max(ATLASradii), 
               min_dist_to_detector_layer = 0.0001, data_combination = 'SM and Signal'):
    if os.path.exists(os.path.join(output_dir, new_output_folder)):
        pass
    else:
        os.mkdir(os.path.join(output_dir, new_output_folder))
    print("making files")
    if (data_combination == 'SM and Signal') or (data_combination == 'SM'):
        bkg_hits = make_bkg_hits(input_dir)
    
    if datatype == 'train':
        datatype_size = train_size
    elif datatype == 'validate':
        datatype_size = validate_size
    elif datatype == 'test':
        datatype_size = test_size

    number_of_chunks = np.floor(datatype_size/chunk_size).astype(int)
    remaining_events_after_chunks = datatype_size % chunk_size
    
    for chunk in range(number_of_chunks):
        # event is used as the iterator within the loop inside combine_SM_and_signal_dfs, so we pass event_id_minus_event into the function and add event to it with each loop iteration
        if datatype == 'train':
            event_id_minus_event = chunk * chunk_size + 1
        elif datatype == 'validate':
            event_id_minus_event = train_size + chunk * chunk_size + 1
        elif datatype == 'test':
            event_id_minus_event = train_size + validate_size + chunk * chunk_size + 1
        
        if data_combination == 'SM and Signal':
            combine_SM_and_signal_dfs(chunk, chunk_size, fourierRadii, fourierDim, times, fourierCenters, Lambda, min_dist_to_detector_layer, 
                                      event_id_minus_event, bkg_hits)
        elif data_combination == 'SM':
            prepare_SM_dfs(chunk, chunk_size, event_id_minus_event, bkg_hits, final_iteration = False, remaining_events_after_chunks = None)
        elif data_combination == 'Signal':
            prepare_signal_dfs(chunk, chunk_size, fourierRadii, fourierDim, times, fourierCenters, Lambda, min_dist_to_detector_layer, 
                               event_id_minus_event, final_iteration = False, signal_hits = None, remaining_events_after_chunks = None)
        print("processed a chunk")

    signal_hits = make_list_of_hits_from_fourier_balls(remaining_events_after_chunks, fourierRadii, fourierDim,times , 
                                                       fourierCenters,Lambda , min_dist_to_detector_layer)
    for event in range(remaining_events_after_chunks):
        if datatype == 'train':
            event_id = event + number_of_chunks * chunk_size + 1
        elif datatype == 'validate':
            event_id = event + train_size + number_of_chunks * chunk_size + 1
        elif datatype == 'test':
            event_id = event + train_size + validate_size + number_of_chunks * chunk_size + 1

        if data_combination == 'SM and Signal':
            combine_SM_and_signal_dfs(chunk, chunk_size, fourierRadii, fourierDim, times, fourierCenters, Lambda, min_dist_to_detector_layer, 
                                      event_id, bkg_hits,final_iteration = True, signal_hits = signal_hits, 
                                      remaining_events_after_chunks = remaining_events_after_chunks)
        elif data_combination == 'SM':
            prepare_SM_dfs(chunk, chunk_size, event_id_minus_event, bkg_hits, final_iteration = True, remaining_events_after_chunks = remaining_events_after_chunks)
        elif data_combination == 'Signal':
            prepare_signal_dfs(chunk, chunk_size, fourierRadii, fourierDim, times, fourierCenters, Lambda, min_dist_to_detector_layer, 
                               event_id, final_iteration = True, signal_hits = signal_hits, 
                               remaining_events_after_chunks = remaining_events_after_chunks)
    del bkg_hits

make_files(input_dir = input_dir, datatype = 'train', signal_tracks_per_event = signal_tracks_per_event,fourierRadii = fourierRadiiTrain,fourierDim = fourierDimTrain, 
          times = times, fourierCenters = fourierCenters, Lambda = Lambda, min_dist_to_detector_layer = min_dist_to_detector_layer, data_combination=data_combination_train)

make_files(input_dir = input_dir, datatype = 'test', signal_tracks_per_event = signal_tracks_per_event,fourierRadii = fourierRadiiTest,fourierDim = fourierDimTest, 
          times = times, fourierCenters = fourierCenters, Lambda = Lambda, min_dist_to_detector_layer = min_dist_to_detector_layer, data_combination =data_combination_test)

#tracks_cylindrical_fourier_balls(times,3, Lambda, 5, fourierRadii, fourierCenters)

# result = make_list_of_hits_from_fourier_balls(5, fourierRadii, 3,times ,fourierCenters ,Lambda = np.max(ATLASradii), min_dist_to_detector_layer = 0.001)
# make_track_plot(result, 2)
# #if __name__ == "main":
