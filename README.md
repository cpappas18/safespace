# safespace
A tool for retailers to ensure their customers have a safe *socially-distanced* shopping experience. 

We built this project for the 2020 McGill Artificial Intelligence Society Hackathon. This tool enables users to analyze video footage of a public retail space to track their current capacity and the extent to which social distancing guidelines are being followed. We built off of existing code from OpenCV which uses object tracking to detect when new objects enter the frame and to see how existing objects traverse the space. We added new features to track real-time distance measurements between occupants and to assess current risk levels. 

Capacity Count:
- Records incoming & outgoing traffic to keep a live count of the current occupancy of the space
- Alerts the user when a pre-determined maximum capacity is exceeded

Distance Calculator:
- Measures the distance between all customers to ensure proper distancing 
- Displays risk status (low/moderate/high) according to the number of customers in close proximity

References: 
https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/
