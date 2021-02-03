# molecule-solution
This is a project for the QSciTech workshop on IBMQ quantum computing. Includes the completed version of the suggested solution to find the ground state of a molecule using quantum computing. All necessary and optional methods have been implemented.

Description of the files :
- hamiltonian.py : Defines the FermionicHamiltonian class and subclasses.
- pauli_string.py : Defines PauliString and LinearCombinaisonPauliString class.
- mapping.py : Defines the JordanWigner mapping and the Parity mapping (optional assignment).
- evaluator.py : Defines the abstract class Evaluator and the BasicEvaluator class as well as the BitwiseCommutingCliqueEvaluator class (optional assignment).
- solve.py : Defines VQESolver and ExactSolver.
 
Other files :
- Integrals_sto-3g_H2_d_0.7350_no_spin.npz : Contains the one body and two body integrals (no spin) for a H2 molecule with d=0.735. The two body is given in the physicist order.
