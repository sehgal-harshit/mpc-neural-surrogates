
reactor --> data sampling --> SS_NARX, MSA_NARX, TiDE --> ~~AR Simulation MPCs (no true readings at all)~~

- heat transfer coeff b/w jacket and reactor, and b/w jacket and env --- time varying uncertainities -- reactor_heat_transfer_coefficient (1), heat_loss_coefficient(2)
- flow_input (3) -- random walk tvp
- Above 3 should vary when base_cobr is simulated alongside surrogate-based-MPC

Surrogate based MPC + base_cobr in parallel ----> 3 case studies

SS_NARX -- no UQ, UQ --- covariance propogation
MSA_NARX --- no UQ, UQ --- CQR
TiDE --- no UQ, UQ --- CQR

-- 6x MPC sims

get the framework ready !

- post training UQ for SS_NARX
- common cqr pipeline for MSA and TiDE


## 23_06 --
 - Sensitivity to uncertain params -- also added in SS_uq 
 - Try the MSA and TiDE models in do_mpc as well
