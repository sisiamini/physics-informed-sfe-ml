# Elemental property constants and descriptor definitions

This document records the elemental properties used to construct the four composition-derived descriptors in the stacking-fault-energy (SFE) modelling dataset. The numerical values are identical to those in `elemental_property_table.csv` and the archived implementation `scripts/01_feature_generation/SFE_with_Descriptors.py`.

## Elemental properties

Atomic weights are expressed in g mol⁻¹, radii in pm, electronegativities on the dimensionless Pauling scale, and valence-electron counts (VEC) as nominal electrons per atom.

| Element | Atomic weight | Atomic radius | Pauling electronegativity | Nominal VEC | Radius convention |
|:--|--:|--:|--:|--:|:--|
| Fe | 55.845 | 126 | 1.83 | 8 | Metallic |
| Mn | 54.938044 | 127 | 1.55 | 7 | Metallic |
| Al | 26.9815385 | 143 | 1.61 | 3 | Metallic |
| C | 12.011 | 70 | 2.55 | 4 | Empirical |
| Ni | 58.6934 | 124 | 1.91 | 10 | Metallic |
| Cr | 51.9961 | 128 | 1.66 | 6 | Metallic |
| Si | 28.085 | 111 | 1.90 | 4 | Calculated/covalent numerical correspondence |
| Cu | 63.546 | 128 | 1.90 | 11 | Metallic |
| Mo | 95.95 | 139 | 2.16 | 6 | Metallic |
| Nb | 92.90637 | 146 | 1.60 | 5 | Metallic |
| V | 50.9415 | 134 | 1.63 | 5 | Metallic |
| Ti | 47.867 | 147 | 1.54 | 4 | Metallic |
| Co | 58.933194 | 125 | 1.88 | 9 | Metallic |
| N | 14.007 | 65 | 3.04 | 5 | Empirical |
| P | 30.973762 | 100 | 2.19 | 5 | Empirical |
| S | 32.06 | 100 | 2.58 | 6 | Empirical |

## Numerical conventions and source correspondence

### Atomic weights

The values are conventional atomic weights used to convert reported alloy compositions from weight percent to atomic fractions. They agree with standard or abridged atomic weights maintained by the Commission on Isotopic Abundances and Atomic Weights (CIAAW) to the tabulated precision or historical rounding. They are not uniformly identical to the latest CIAAW digits: for example, the archived Al and Mn values differ slightly from the 2024 table, while C, N, Si and S have interval standard atomic weights for normal terrestrial materials. The archived numerical values are retained to preserve exact reproducibility [1].

### Electronegativity

All 16 electronegativity entries correspond numerically to the Pauling-scale values in the PubChem periodic-table compilation [2]. The Pauling electronegativity concept is described in the original work of Pauling [3]. These values are empirical reference properties, rather than measurements on the specific alloy specimens.

### Atomic radii

The archived atomic-radius column is **not a single-definition radius dataset**. The values for Fe, Mn, Al, Ni, Cr, Cu, Mo, Nb, V, Ti and Co coincide with a documented twelve-coordinate metallic-radius dataset attributed to Greenwood and Earnshaw [4]. The values C = 70, N = 65, P = 100 and S = 100 pm correspond to empirical atomic radii associated with Slater's crystal-radius compilation [5]. The Si value of 111 pm is also found in calculated-radius compilations and in the single-bond covalent-radius set of Cordero et al. [6]. These numerical correspondences are retrospective; they do not establish the original historical source from which the archived array was assembled.

Metallic, empirical and covalent radii represent different reference conventions. Therefore, `R_mean (pm)` and `R_std (pm)` are fixed-reference, composition-weighted numerical descriptors. They are not direct measurements of atomic size, lattice distortion or substitutional size mismatch within the experimental steels. All 16 elements, including C and N, enter the archived radius calculations. A separate, consistently defined covalent-radius sensitivity dataset is maintained only as an alternative modelling experiment; it does not replace the archived constants.

### Valence-electron count

VEC is defined by nominal periodic-group counts for the elements represented in the dataset. For transition metals, the adopted values correspond to nominal group numbers (e.g., Fe = 8, Ni = 10, Cu = 11); for main-group elements they correspond to common valence counts (e.g., C = 4, N = 5, S = 6). Composition-weighted VEC is a recognized alloy descriptor [7]. Extending the same arithmetic convention to interstitial and minor non-metal elements in steel is a modelling definition, not an experimental determination of their electronic states.

## Descriptor calculation

Let `w_i` be elemental composition in wt.%, `A_i` the tabulated atomic weight, `r_i` the archived radius, `χ_i` the Pauling electronegativity and `v_i` the nominal VEC. The atomic fraction of element `i` is

\[
 x_i=\frac{w_i/A_i}{\sum_j w_j/A_j}.
\]

The four descriptors are

\[
\begin{aligned}
 R_{\mathrm{mean}} &= \sum_i x_i r_i,\\
 R_{\mathrm{std}} &= \sqrt{\sum_i x_i(r_i-R_{\mathrm{mean}})^2},\\
 \overline{\chi} &= \sum_i x_i \chi_i,\\
 \Delta\chi &= \sqrt{\sum_i x_i(\chi_i-\overline{\chi})^2},\\
 \mathrm{VEC}_{\mathrm{mean}} &= \sum_i x_i v_i.
\end{aligned}
\]

The descriptor calculations use all 16 composition variables, including any elements present at zero weight percent. The descriptors are deterministic functions of composition and do not constitute independent experimental observations. The exact baseline arrays and implementation are preserved in `elemental_property_table.csv` and `SFE_with_Descriptors.py`.

## References

1. Commission on Isotopic Abundances and Atomic Weights. (2024). *Standard atomic weights 2024*. International Union of Pure and Applied Chemistry. https://ciaaw.org/atomic-weights.htm ; *Abridged standard atomic weights 2024*. https://ciaaw.org/abridged-atomic-weights.htm
2. National Center for Biotechnology Information. (n.d.). *Electronegativity in the periodic table of elements (Pauling scale)*. PubChem. https://pubchem.ncbi.nlm.nih.gov/periodic-table/electronegativity
3. Pauling, L. (1932). The nature of the chemical bond. IV. The energy of single bonds and the relative electronegativity of atoms. *Journal of the American Chemical Society, 54*(9), 3570–3582. https://doi.org/10.1021/ja01348a011
4. Greenwood, N. N., & Earnshaw, A. (1997). *Chemistry of the elements* (2nd ed.). Butterworth-Heinemann. Matching metallic-radius entries and their attribution to this reference are reproduced in the AmorphGen scientific-software data table: https://smtg-bham.github.io/AmorphGen/_modules/amorphgen/utils/radii.html . The original textbook entries were not independently verified page by page.
5. Slater, J. C. (1964). Atomic radii in crystals. *The Journal of Chemical Physics, 41*(10), 3199–3204. https://doi.org/10.1063/1.1725697
6. Cordero, B., Gómez, V., Platero-Prats, A. E., Revés, M., Echeverría, J., Cremades, E., Barragán, F., & Alvarez, S. (2008). Covalent radii revisited. *Dalton Transactions*, (21), 2832–2838. https://doi.org/10.1039/B801115J
7. Guo, S., Ng, C., Lu, J., & Liu, C. T. (2011). Effect of valence electron concentration on stability of fcc or bcc phase in high entropy alloys. *Journal of Applied Physics, 109*(10), 103505. https://doi.org/10.1063/1.3587228
