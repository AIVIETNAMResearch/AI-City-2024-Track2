import sys
sys.path.append('../utils')
from utils import load_json, save_json, fill_video, longest

print('Perform postprocessing')
# External
pedes_rewrite_external = load_json('../../aux_dataset/results/test/external/pedes_rewrite_50%.json')

vehicle_rewrite_external = {}
vehicle_segment_external = {}
for segment_type in ['appearance', 'environment', 'location', 'action']:
    vehicle_segment_external[segment_type] = load_json(f"../../aux_dataset/results/test/external/vehicle_{segment_type}.json")
    
for video_name in pedes_rewrite_external.keys():
    vehicle_video_result = dict()
    for phase_number in range(5):
        phase_number = str(phase_number)
        if vehicle_segment_external[segment_type][video_name][phase_number] is not None:
            tmp = []
            for segment_type in ['action', 'location', 'appearance', 'environment']:
                tmp.append(vehicle_segment_external[segment_type][video_name][phase_number][segment_type])
            tmp = ' '.join(tmp)
            vehicle_video_result[phase_number] = dict(caption_vehicle=tmp)
        else:
            vehicle_video_result[phase_number] = None
            
    try:
        pedes_rewrite_external[video_name] = fill_video(pedes_rewrite_external[video_name])
        vehicle_rewrite_external[video_name] = fill_video(vehicle_video_result)
    except:
        pedes_rewrite_external[video_name] = None
        vehicle_rewrite_external[video_name] = None
        print(video_name) # This video have none bbox annotations
    
save_json('../../aux_dataset/submission/pedes_external.json', pedes_rewrite_external)
save_json('../../aux_dataset/submission/vehicle_external.json', vehicle_rewrite_external)

# Internal
# Vehicle View
pedes_rewrite_internal_v = load_json('../../aux_dataset/results/test/internal/vehicle_view/pedes_rewrite.json')
vehicle_rewrite_internal_v = load_json('../../aux_dataset/results/test/internal/vehicle_view/vehicle_rewrite.json')
for video_name in pedes_rewrite_internal_v.keys():
    for phase_number in range(5):
        phase_number = str(phase_number)
        if phase_number not in pedes_rewrite_internal_v[video_name]:
            pedes_rewrite_internal_v[video_name][phase_number] = None
            
        if phase_number not in vehicle_rewrite_internal_v[video_name]:
            vehicle_rewrite_internal_v[video_name][phase_number] = None
    
    try:
        pedes_rewrite_internal_v[video_name] = fill_video(pedes_rewrite_internal_v[video_name])
        vehicle_rewrite_internal_v[video_name] = fill_video(vehicle_rewrite_internal_v[video_name])
    except:
        pedes_rewrite_internal_v[video_name] = None
        vehicle_rewrite_internal_v[video_name] = None
        print(video_name) # This video have none bbox annotations

save_json('../../aux_dataset/submission/pedes_internal_v.json', pedes_rewrite_internal_v)
save_json('../../aux_dataset/submission/vehicle_internal_v.json', vehicle_rewrite_internal_v)

# Overhead View
pedes_rewrite_internal_o = load_json('../../aux_dataset/results/test/internal/overhead_view/pedes_rewrite.json')
vehicle_rewrite_internal_o = load_json('../../aux_dataset/results/test/internal/overhead_view/vehicle_rewrite.json')
for video_name in pedes_rewrite_internal_o.keys():
    pedes_rewrite_internal_o[video_name] = longest(pedes_rewrite_internal_o[video_name], 'caption_pedestrian')
    vehicle_rewrite_internal_o[video_name] = longest(vehicle_rewrite_internal_o[video_name], 'caption_vehicle')

    for phase_number in range(5):
        phase_number = str(phase_number)
        if phase_number not in pedes_rewrite_internal_o[video_name]:
            pedes_rewrite_internal_o[video_name][phase_number] = None
            
        if phase_number not in vehicle_rewrite_internal_o[video_name]:
            vehicle_rewrite_internal_o[video_name][phase_number] = None

    try:
        pedes_rewrite_internal_o[video_name] = fill_video(pedes_rewrite_internal_o[video_name])
        vehicle_rewrite_internal_o[video_name] = fill_video(vehicle_rewrite_internal_o[video_name])
    except:
        pedes_rewrite_internal_v[video_name] = None
        vehicle_rewrite_internal_v[video_name] = None
        print(video_name) # This video have none bbox annotations
        
save_json('../../aux_dataset/submission/pedes_internal_o.json', pedes_rewrite_internal_o)
save_json('../../aux_dataset/submission/vehicle_internal_o.json', vehicle_rewrite_internal_o)