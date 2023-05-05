# Introduction
This project is about finding Camera Callibration(K) from 3 pair of vanishing points which are orthogonal.

![q2a_annotated](https://user-images.githubusercontent.com/22910010/236413072-7cca1404-c1c0-4124-b03a-709d9aa3d826.png)

# How to Use:

## Annotate Image

![image_anotated](https://user-images.githubusercontent.com/22910010/236413453-549afa01-1230-4ca0-90bb-c5347d487c18.png)

Annotate the input image by running `python3 main.py` and then select the 3 pairs of orthogonal vanishing points.
This will generate 'final_cord.npy'.

You can skip this step and directly procede to run the program as 'final_cord.npy' is already present.

## Run the program

run the program by executing `python3 main.py`.


# Output

```
Intrinsic Matrix: [[841.11411487   0.         587.63623557]
 [  0.         841.11411487 331.75406136]
 [  0.           0.           1.        ]] 

```

# Conclusion

