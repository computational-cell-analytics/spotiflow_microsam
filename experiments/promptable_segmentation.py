import os
from glob import glob
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import imageio.v3 as imageio
from skimage.measure import regionprops
from skimage.measure import label as connected_components

from micro_sam.prompt_based_segmentation import segment_from_points
from micro_sam.util import get_sam_model, precompute_image_embeddings

from segment_anything import SamPredictor


ROOT = "/home/anwai/Downloads/spotiflow_187k_octavio_test"


def run_promptable_segmentation(
    predictor: SamPredictor, image: np.ndarray, point_prompts: List[List[Tuple[int, int]]],
) -> np.ndarray:
    """Functionality to run promptable semgentation using generated point prompts.

    Args:
        predictor: The Segment Anything model to generate predictions with input prompts.
        image: The input image.
        point_prompts: The positive point prompts to generate masks.

    Returns:
        The instance segmentation.
    """

    # Compute the image embeddings.
    image_embeddings = precompute_image_embeddings(
        predictor=predictor,
        input_=image,
        ndim=2,  # With RGB images, we should have channels last and must set ndim to 2.
        verbose=False,
        tile_shape=(384, 384),  # The tile shape for larger images to perform prediction using tiling-window.
        halo=(64, 64),  # The overlap shape for larger images to perform prediction using tiling-window.
        # save_path=f"embeddings_{i}.zarr",  # Stores the image embeddings locally, otherwise computes on-the-fly.
    )

    # Run promptable segmentation.
    masks = [
        segment_from_points(
            predictor=predictor,
            points=np.array([each_point_prompt]),  # Each point coordinate (Y, X) is expected as array.
            labels=np.array([1]),  # Each corresponding label, eg. 1 corresponds positive, is expected as array.
            image_embeddings=image_embeddings,
        ).squeeze() for each_point_prompt in point_prompts
    ]

    # Merge all masks into one segmentation.
    # 1. First, we get the area per object and try to map as: big objects first and small ones then
    #    (to avoid losing tiny objects near-by or to any possible overlaps)
    mask_props = [{"mask": mask, "area": regionprops(connected_components(mask))[0].area} for mask in masks]

    # 2. Next, we assort based on area from greatest to smallest.
    assorted_masks = sorted(mask_props, key=(lambda x: x["area"]), reverse=True)
    masks = [per_mask["mask"] for per_mask in assorted_masks]

    # 3. Finally, we merge all individual segmentations into one.
    segmentation = np.zeros(image.shape[:2], dtype=int)
    for j, mask in enumerate(masks, start=1):
        segmentation[mask > 0] = j

    return segmentation


def main():

    for p in glob(os.path.join(ROOT, "*.tif")):

        # Read the image
        image = imageio.imread(p)

        # Load the prompts
        csv_path = Path(p).with_suffix(".csv")
        df = pd.read_csv(csv_path)

        # Get the exact point prompts
        point_prompts = [(r[1]["y"], r[1]["x"]) for r in df.iterrows()]

        # Perform promptable segmentation
        predictor = get_sam_model(model_type="vit_b_lm")
        segmentation = run_promptable_segmentation(
            predictor=predictor, image=image, point_prompts=point_prompts
        )

        # Visualize the results
        import napari
        v = napari.Viewer()
        v.add_image(image)
        v.add_labels(segmentation)
        v.add_points(point_prompts)
        napari.run()


if __name__ == "__main__":
    main()
