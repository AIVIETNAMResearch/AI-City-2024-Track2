import sys
sys.path.append('../utils')
from utils import load_json, save_json

print('Perform generate submission')

# Load submission template
submission = load_json('../../aux_dataset/submission_template.json')

# External
pedes_external = load_json('../../aux_dataset/submission/pedes_external.json')
vehicle_external = load_json('../../aux_dataset/submission/vehicle_external.json')

for video_name in pedes_external.keys():
    video_result = []
    if pedes_external[video_name] is None:
        continue
    
    for phase_number in range(5):
        phase_number = str(phase_number)
        video_result.append({
            "labels": [
                phase_number
            ],
            "caption_pedestrian": pedes_external[video_name][phase_number]['caption_pedestrian'],
            "caption_vehicle": vehicle_external[video_name][phase_number]['caption_vehicle']
        })
    submission[video_name] = video_result
    
# Internal
# Vehicle View
pedes_internal_v = load_json('../../aux_dataset/submission/pedes_internal_v.json')
vehicle_internal_v = load_json('../../aux_dataset/submission/vehicle_internal_v.json')

# Overhead View
pedes_internal_o = load_json('../../aux_dataset/submission/pedes_internal_o.json')
vehicle_internal_o = load_json('../../aux_dataset/submission/vehicle_internal_o.json')

for video_name in pedes_internal_v.keys():
    video_result = []
    
    if pedes_internal_v[video_name] is None:
        continue
    
    for phase_number in range(5):
        phase_number = str(phase_number)
        try:
            video_result.append({
                "labels": [
                    phase_number
                ],
                "caption_pedestrian": pedes_internal_v[video_name][phase_number]['caption_pedestrian'],
                "caption_vehicle": vehicle_internal_o[video_name][phase_number]['caption_vehicle']
            })
        except:
            video_result.append({
                "labels": [
                    phase_number
                ],
                "caption_pedestrian": pedes_internal_v[video_name][phase_number]['caption_pedestrian'],
                "caption_vehicle": vehicle_internal_v[video_name][phase_number]['caption_vehicle']
            })
    submission[video_name] = video_result

for video_name in pedes_internal_o.keys():
    video_result = []
    
    if pedes_internal_o[video_name] is None:
        continue
    
    if submission[video_name][0]['caption_pedestrian'] != "":
        continue
    
    for phase_number in range(5):
        phase_number = str(phase_number)
        video_result.append({
            "labels": [
                phase_number
            ],
            "caption_pedestrian": pedes_internal_o[video_name][phase_number]['caption_pedestrian'],
            "caption_vehicle": vehicle_internal_o[video_name][phase_number]['caption_vehicle']
        })
    submission[video_name] = video_result

prev_video = None
for video_name in submission.keys():
    if submission[video_name][0]['caption_pedestrian'] == "":
        submission[video_name] = prev_video
    else:
        prev_video = submission[video_name]

save_json('../../aux_dataset/submission.json', submission)