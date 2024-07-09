# Bulk-Surface Wave-pinning model of Cell Polarisation

This is a bulk-surface FEM code based on the paper by [Cussedu](https://doi.org/10.1016/j.jtbi.2018.09.008) *et al*.
The numerical scheme of equations 41-43 in the paper are supposed to be conservative. But, as per my calculations, they are not conservative. I have re-calculated a conservative scheme which is presented in Equations.pdf file.

We have used [deal.ii](https://www.dealii.org) software package to implement the finite element analysis.
