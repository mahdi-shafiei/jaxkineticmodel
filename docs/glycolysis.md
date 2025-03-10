# Custom ODE models
Not all kinetic model can be drafted only from stoichiometry. There are empirical laws that might influence 
There are many empirical laws that might influence reactions in the system of ODEs. 
Below, we present a reimplemented version of a glycolysis model[^1]
in Jax. We wil note down specific modeling choices that can be used to incorporate 
non-stoichiometric modifications. This output of this script can be found
[SBML model](https://github.com/AbeelLab/jaxkineticmodel/blob/main/models/manual_implementations/sbml_export/glycolysis_feastfamine_pulse1.xml)

## Rate laws
We start by importing relevant modules from `jaxkineticmodel`.

```python 
{!code/glycolysis.py!lines=1-17}
```
We define all reactions that are in the glycolysis model by using the `Reaction` object.
This requires setting a name, species that are in the stoichiometry, the stoichiometry itself, 
and the mechanism used to compute. Note that metabolites (species) that only 'modify' a reaction,
that is, they are involved in the reaction but not consumed or produced, should
also be included in species list and stoichiometry as 0. 


```python
{!code/glycolysis.py!lines=19-83}
```

The next mechanism is an example of adding a modifier structure to a pre-existing mechanism.
This multiplies the calculated `v` by the modifier construct.

```python
{!code/glycolysis.py!lines=83-102}
```

Some other reactions

```python
{!code/glycolysis.py!lines=102-343}
```

Sometimes, a flux may need to be scaled differently per ODEs. Suppose we 
have the original mass-balances in the glycolysis model, 
where for the intracellular concentration we do not scale the flux, but for the
extracellular we do.

$$\frac{d etoh_{ic}}{dt}=- v_{ETOHT} + ... $$
$$\frac{d etoh_{ec}}{dt}=+ v_{ETOHT} * \[biomass\] * 0.002 + ... $$

We can first define a mechanisms for ICETOH and then modify this for ECETOH.
This requires us to use two `Reaction` objects, and set stoichiometry to 0 to 
ensure that they are not wrongly affected by the scaling.

```python
{!code/glycolysis.py!lines=343-365}
```

Further finish the reaction definitions

```python
{!code/glycolysis.py!lines=365-485}
```

## Setting up the ODE system
We now pass all reactions and compartment definitions to `JaxKineticModelBuild`, 
similarly to how it was performed in the Building Models tutorial.

```python
{!code/glycolysis.py!lines=485-505}
```

## Cubic spline boundary condition
As mentioned in the Building Models tutorial, we can add boundary conditions for 
species that are either constant or time-dependent. Often, it is necessary 
to do interpolation, such as cubic spline, to model a perturbation. We can 
build expressions for that using sympy, pass these as a string input
to the boundary condition class, which builds the cubic spline for us, that can
be exported to an sbml.

```python
{!code/glycolysis.py!lines=506-518}
```

## Parameters and initial conditions
We add parameters from literature values and the initial conditions 
are initialized.

```python
{!code/glycolysis.py!lines=520-583}
```

## Exporting the model
The model can be exported to sbml format for parameter optimziation. 

```python
{!code/glycolysis.py!lines=583-593}
```

## References
[^1]: Lao-Martil, D., Schmitz, J. P., Teusink, B., & van Riel, N. A. (2023). Elucidating yeast glycolytic dynamics at steady state growth and glucose pulses through kinetic metabolic modeling. Metabolic engineering, 77, 128-142.

