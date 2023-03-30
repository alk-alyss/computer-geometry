## Lab 02 - Convex Hulls

#Environment setup

## PRELIMINARY
-> Open up conda terminal
-> conda create --name compgeo python=3.9
-> open vscode
-> create folder, download the lab files
-> activate environment
-> install numpy, open3d and scipy
    -> pip install numpy
    -> pip install scipy
    -> pip install open3d
-> run the main.py file

#Convex Hulls

## TASK 1
-> Go over graham scan theory step by step, explaining the tools we will use for the implementation
    -> direction, based on cross-product
    -> explain the angle-based sorting
    -> write the loop
    -> check results
##  -> time?

## TASK 2
-> Go over quickhull theory step by step
    -> furthest point from line
    -> split points by line
    -> keep track of direction vectors
    -> complete function
    -> check results
##  -> time?


-> Visualize both 



#Homework

##  -> Complete the jarvis/gift wrapping algorithm (task 3)
##  -> Select a point on the plane (task 4)
##  -> Paint red all edges that can be seen from that point and blue otherwise
##  -> Compare the 3 algorithms in speed. What's the computational complexity of each one? 
##  -> visualize the amount of time it takes, with respect to the amount of input points
##  -> explain in your report which is better and why