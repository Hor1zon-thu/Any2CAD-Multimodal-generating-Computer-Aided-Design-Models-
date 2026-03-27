import cadquery as cq
import numpy as np
import os
import json
import argparse
from typing import Tuple, Union
import shutil
from tqdm import tqdm

def cq_align_shapes(source : cq.Workplane, target : cq.Workplane) -> Tuple[cq.Workplane, float]:
    """Align source to target using the center of mass and the principal axes of inertia. also return normalized IOU"""
    c_source = cq.Shape.centerOfMass(source.val())
    c_target = cq.Shape.centerOfMass(target.val())

    I_source = np.array(cq.Shape.matrixOfInertia(source.val()))
    I_target = np.array(cq.Shape.matrixOfInertia(target.val()))

    v_source = cq.Shape.computeMass(source.val())
    v_target = cq.Shape.computeMass(target.val())

    I_p_source, I_v_source = np.linalg.eigh(I_source)
    I_p_target, I_v_target = np.linalg.eigh(I_target)

    s_source = np.sqrt(np.abs(I_p_source).sum()/v_source)
    s_target = np.sqrt(np.abs(I_p_target).sum()/v_target)

    normalized_source = source.translate(-c_source).val().scale(1/s_source)
    normalized_target = target.translate(-c_target).val().scale(1/s_target)

    Rs = np.zeros((4,3,3))
    Rs[0] = I_v_target @ I_v_source.T

    for i in range(3):
        # all possible 2 out of 3 permutations
        alignment = 1 - 2 * np.array([i>0, (i+1)%2, i%3<=1])
        Rs[i+1] = I_v_target @ (alignment[None,:] * I_v_source).T

    best_IOU = 0.0
    best_T = None
    for i in range(4):
        T = np.zeros([4,4])
        T[:3,:3] = Rs[i]
        T[-1,-1] = 1
        
        aligned_source = normalized_source.transformGeometry(cq.Matrix(T.tolist()))
        
        try:
            intersect = aligned_source.intersect(normalized_target)
            union = aligned_source.fuse(normalized_target)
            
            IOU = intersect.Volume() / union.Volume()
        except: #handle cases where IOU is undefined
            IOU = 0.0
        
        if IOU > best_IOU:
            best_IOU = IOU
            best_T = T

    if best_T is not None:
        aligned_source = normalized_source.transformGeometry(cq.Matrix(best_T.tolist())).scale(s_target).translate(c_target)
        return cq.Workplane(aligned_source), best_IOU, c_source, c_target
    else:
        aligned_source = None
        return aligned_source, best_IOU, c_source, c_target

def find_image_by_question_id(jsonl_path, target_question_id):
    with open(jsonl_path, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)  # Parse JSON line
            if data.get("question_id") == target_question_id:
                return data.get("image")[:-6]  # Return the "image" field
    return None  # Return None if not found

def average_non_none(values):
    filtered_values = [v for v in values if v is not None]
    print(f"Number of Nones: {len(values) - len(filtered_values)}")
    return sum(filtered_values) / len(filtered_values) if filtered_values else None


def main(gen_dir, gt_dir, jsonl_path=None):
    model_generated_steps_dir = gen_dir
    ground_truth_generated_steps_dir = gt_dir
    
    if not os.path.exists(model_generated_steps_dir):
        print(f"Error: Generated steps directory not found: {model_generated_steps_dir}")
        return
    if not os.path.exists(ground_truth_generated_steps_dir):
        print(f"Error: Ground truth directory not found: {ground_truth_generated_steps_dir}")
        return

    # Ensure directories end with /
    if not model_generated_steps_dir.endswith("/"): model_generated_steps_dir += "/"
    if not ground_truth_generated_steps_dir.endswith("/"): ground_truth_generated_steps_dir += "/"

    all_ious = []
    gt_steps = []
    model_steps = []
    model_steps_aligned = []
    
    files = [f for f in os.listdir(model_generated_steps_dir) if f.endswith(".step")]
    
    # Add tqdm to below
    for g in tqdm(files):
        # print(f"Processing: {g}")
        question_id = g[:-5]
        
        if jsonl_path:
            orig_id = find_image_by_question_id(jsonl_path, int(question_id))
            if orig_id == None:
                print(f"Warning: Can't find original ID for {question_id} in jsonl")
                continue
        else:
            orig_id = question_id # Assume filenames match if no jsonl provided
        
        gt_path = ground_truth_generated_steps_dir + orig_id + ".step"
        if not os.path.exists(gt_path):
            # Try checking if gt filename needs extension or not, or if it differs
            # For now assume extension is needed
            print(f"Warning: GT file not found: {gt_path}")
            continue

        try:
            gt_step = cq.importers.importStep(gt_path)
            model_generated_step = cq.importers.importStep(model_generated_steps_dir + g)
        except Exception as e:
            print(f"Error loading STEP files for {g}: {e}")
            continue

        gt_steps.append(gt_step)
        model_steps.append(model_generated_step)
        try:
            aligned_model_generated, IOU, _, _ = cq_align_shapes(model_generated_step, gt_step)
            model_steps_aligned.append(aligned_model_generated)
            all_ious.append(IOU)
        except: #fix:only gemini seemed to have this problem, added try except statement for gemini
            print(f"Issue computing IOU for {g}")

    avg_iou = average_non_none(all_ious)
    print(f"Model's average IoU score: {avg_iou}")
    
    iou_result_file = os.path.join(model_generated_steps_dir, "cad_iou_results.txt")
    with open(iou_result_file, "w", encoding="utf-8") as f:
        f.write(f"Generated Dir: {model_generated_steps_dir}\n")
        f.write(f"GT Dir: {ground_truth_generated_steps_dir}\n")
        f.write(f"Average IoU: {avg_iou}\n")
        f.write(f"Number of valid steps: {len(all_ious)}\n")
    print(f"Results saved to {iou_result_file}")
    return

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Compute model's IoU score.")
    parser.add_argument("--gen_dir", type=str, default="scripts/test", help="Directory containing generated STEP files.")
    parser.add_argument("--gt_dir", type=str, default="scripts/gt", help="Directory containing Ground Truth STEP files.")
    parser.add_argument("--jsonl_path", type=str, default=None, help="Optional JSONL path for mapping IDs.")
    
    args = parser.parse_args()

    main(args.gen_dir, args.gt_dir, args.jsonl_path)