# pyvisionfilters
Linear and non-linear filters for simple computer vision tasks

#### visionfilters.py

```python
# Return a box filter of size n by n
# Requires n to be an odd integer
boxfilter(n)

# Return a normalized 1D Gaussian filter
# Fix length of filter array to (6 * sigma) rounded up to the next odd integer
gauss1d(sigma)
```
