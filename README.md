Takes a path and produces a triangle mesh that corresponds to the antialiased stroked path.

The approach here is naive and only works for opaquely filled paths. Overlaping areas can 
end up with seams or otherwise incorrect coverage values.
