# TER2021-074 - Deep Learning for Autonomous Cars / Federated learning, communication inter-objets

## 🏫 Our group
* Hadrien BONATO-PAPE (SI5-WIA)
* Hossein FIROOZ (M2-EIT Digital)
* Vincent LAUBRY (M2-IoTCPS)
* Erdal TOPRAK (M2-SD)


## Objectives

Today, ADAS systems such as parking assistance or lane centering are among the functionalities that are commonly available on a car. To remain competitive, car manufacturers must provide best-in-class ADAS systems. Some ADAS systems consist of two main steps: 
·       Detection that aims to extract certain characteristics (track, edges, center line, traffic signs, etc.) by analyzing the signals picked up by sensors (radar, camera, etc.). 
·       Followed by a decision stage that allows to control the dynamics of the car. 

By applying these 2 steps in the case of lane centering and traffic sign detection, the steering wheel angle is automatically adjusted according to the lane detection, and the speed is impacted by the traffic signs detected. 
One of the main challenges is to determine how to embed efficient neural networks in low power platforms while preserving performances. 

Objectives: 
1. Understand tricks to better embed deep neural networks on small platforms such as Raspberry Pi. 

2. Combine several models (one for path following, one for road signs recognition, one for pedestrian detection) and design an "orchestrator" (this corresponds to Federated Learning problem) to build a meta-decision on when to check which model. 

3. Install and use a fake GPS with a  wide-angle camera filming from the top (from the ceiling) 

All this will be tested at the Maison de l'Intelligence Artificielle, on a real track with real mini-cars.

## Libraries used
Flower
PyTorch
OpenCV

## [🗂 Documentation](https://github.com/erdaltoprak/TER2021-074/tree/master/Documentation)
Find task implementations in the Code folder, task descriptions in Documentation folder and the final report in Documantations/final_report.pdf

