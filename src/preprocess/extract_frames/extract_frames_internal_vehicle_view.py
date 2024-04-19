import os
import cv2
import json
import ffmpeg
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from os.path import join as osp 

mapper = {
    "prerecognition":"0",
    "recognition":"1",
    "judgement":"2",
    "action":"3",
    "avoidance":"4"
    }

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def main(args):
    type = args.type
    root = '../../../dataset'
    root_output_dir = '../../../aux_dataset/extracted_frames/internal/vehicle_view'
    
    # Create save directory
    os.makedirs('../../../aux_dataset/extracted_frames/internal', exist_ok=True)        
    
    if type == 'all':
        types = ['train', 'val', 'test']
    else:
        types = [type]
    
    for type in types:
        print(f"Start extracting video frames for: {type} set")
        
        anno_root = osp(root, 'annotations')
        video_roots = [osp(root, f'videos/{type}'), osp(root, f'videos/{type}/normal_trimmed')]
        caption_anno_roots = [osp(anno_root, f'caption/{type}'), osp(anno_root, f'caption/{type}/normal_trimmed')]
        bbox_anno_roots = [osp(anno_root, f'bbox_annotated', 'pedestrian', type), osp(anno_root, f'bbox_annotated', 'pedestrian', type, 'normal_trimmed')]
        output_dir = osp(root_output_dir, type)

        for caption_anno_root, bbox_anno_root, video_root in zip(caption_anno_roots, bbox_anno_roots, video_roots):
            video_paths = os.listdir(video_root)

            for video_name in tqdm(video_paths):
                caption_anno =  osp(caption_anno_root, video_name, 'vehicle_view', video_name) + '_caption.json'
                with open(caption_anno, 'r') as f:
                    caption_anno = json.load(f)

                try:
                    with open(osp(bbox_anno_root, video_name, 'vehicle_view', video_name + '_vehicle_view_bbox.json'), 'r') as f:
                        bbox = json.load(f)['annotations']
                except:
                    continue

                bbox_dict = dict()
                for bb in bbox:
                    phase_number = str(bb['phase_number'])
                    bbox_dict[phase_number] = bb
                
                video_path = osp(video_root, video_name, 'vehicle_view', video_name + '_vehicle_view.mp4')
                fps = float(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS))

                phase_captions = dict()
                for e in caption_anno['event_phase']:
                    phase_number = str(e['labels'][0])
                    if bbox_dict.get(phase_number, False):
                        phase_captions[phase_number] = dict(start_time=float(e['start_time']),
                                                            end_time=float(e['end_time']),
                                                            bbox=bbox_dict[phase_number]['bbox'],
                                                            frame_id=bbox_dict[phase_number]['image_id'])
                    else:
                        phase_captions[phase_number] = None

                os.makedirs(osp(output_dir, video_name), exist_ok=True)
                for phase_number, phase_anno in phase_captions.items():
                    try:
                        vcap = cv2.VideoCapture(video_path) # 0=camera
                        width  = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
                        height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`

                        out, _ = (
                            ffmpeg
                            .input(video_path, ss=phase_anno['frame_id']/fps)
                            .output('pipe:', format='rawvideo', pix_fmt='rgb24', vframes=1, loglevel="quiet")
                            .run(capture_stdout=True)
                        )

                        frame = np.frombuffer(out, np.uint8)
                        frame = np.copy(frame.reshape([height, width, 3]))
                        plt.imsave(osp(output_dir, video_name, str(phase_anno['frame_id']) + '.png'), frame)
                    except:
                        print(f"Error in extracting {video_name} at frame {str(phase_anno['frame_id'])}.")
    print(f"End extracting video frames for: {type} set")    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=['train', 'val', 'test', 'all'], default='all', help='Subset choices')
    args = parser.parse_args()
    main(args)