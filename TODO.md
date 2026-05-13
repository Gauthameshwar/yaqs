# List of TODO and next implementations

- [ ] Consider an XXZ tf model with $J \neq \Delta$. This breaks the symmetries that exist in the XXX model, and lets the trajectories explore the full phase space. Reduce the $\Delta$ to smaller value (like 0.7) so that interaction does not restrict the moving of the trajectories in the phase space. 

- [ ] Do a full summation test where you evolve each basis in parallel in the initial state. This will evolve every single basis with time, and it SHOULD lead to the exact evolution. 

- [ ] Verifying the full summation works is a proof that things work as expected with the tensor jump statistical averaging. 

- [ ] Typicality should appear in the system much quicker as you increase the system size, than when you increase the number of trajectories in the ensemble evolution. 

- [ ] Try the test where you have a truncation on the bond dimension and that still offers accuracy coinciding with the ED results. If this works, it is a clear green signal to go ahead and try out the typicality of randomly initialised MPS and evolve them with our TJM with truncated bond dims. 

- [ ] Avoid doing spin-1 systems for now (only for the far future)

- [ ] 