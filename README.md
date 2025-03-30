# AI-Duck-Detector-
CNN-based object detector and regressor for the a task, part of the NitroNLP Hackathon.

This project implements a custom deep learning pipeline for visual duck detection:

Stage 1: A ResNet-based model predicts whether a duck is present and regresses the bounding box coordinates. (2 heads)

Stage 2: A Pixel Refiner model takes the cropped bounding box region and the initial pixel count estimate (via thresholding) and outputs a refined pixel count.

The goal is to handle noisy, varied duck appearances (rotated, resized, flipped) in a controlled evaluation setting.
