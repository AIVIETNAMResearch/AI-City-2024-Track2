import os
import cv2
import sys
import ffmpeg
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from os.path import join as osp 
sys.path.append('../../utils')
from utils import phase2num, load_json

def main(args):
    type = args.type
    root = '../../../dataset/external/BDD_PC_5K'
    root_output_dir = '../../../aux_dataset/extracted_frames/external'
    
    # Create save directory
    os.makedirs(root_output_dir, exist_ok=True)        
    
    if type == 'all':
        types = ['train', 'val', 'test']
    else:
        types = [type]
    
    for type in types:
        print(f"Start extracting video frames for: {type} set")
        
        anno_root = osp(root, 'annotations')
        video_root = osp(root, f'videos/{type}')
        caption_anno_root = osp(anno_root, f'caption/{type}')
        bbox_anno_root = osp(anno_root, f'bbox_annotated/{type}')
        
        output_dir = osp(root_output_dir, type)
        video_paths = sorted(os.listdir(video_root), key=lambda x: int(x[5:][:-4]))

        for video_path in tqdm(video_paths):
            video_name = video_path[:-4]
            caption_anno = load_json(osp(caption_anno_root, video_name) + '_caption.json')
            bounding_box_anno = load_json(osp(bbox_anno_root, video_name) + '_bbox.json')['annotations']

            # Process phase bounding box
            bounding_box_dict = dict()
            for bb in bounding_box_anno:
                phase_number = str(bb['phase_number'])
                bounding_box_dict[phase_number] = bb

            # Process phase annotation
            phase_captions = dict()
            for e in caption_anno['event_phase']:
                if e['labels'][0].isdigit():
                    phase_number = e['labels'][0]
                else:
                    phase_number = phase2num[e['labels'][0]]
                
                if bounding_box_dict.get(phase_number, False):
                    phase_captions[phase_number] = dict(start_time=float(e['start_time']),
                                                        end_time=float(e['end_time']),
                                                        bbox=bounding_box_dict[phase_number]['bbox'],
                                                        frame_id=bounding_box_dict[phase_number]['image_id'])
            
            video_path = osp(video_root, video_path)
            fps = float(caption_anno['fps'])
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
                    frame = frame.reshape([height, width, 3])
                    plt.imsave(osp(output_dir, video_name, str(phase_anno['frame_id']) + '.png'), frame)
                except:
                    print(f"Error in extracting {video_name} at frame {str(phase_anno['frame_id'])}.")
    print(f"End extracting video frames for: {type} set")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=['train', 'val', 'test', 'all'], default='all', help='Subset choices')
    args = parser.parse_args()
    main(args)