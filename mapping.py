"""
mapping.py - Map a Hamiltonian to a LinearCombinaisonPauliString

Copyright 2020-2021 Maxime Dion <maxime.dion@usherbrooke.ca>
This file has been modified by <Your,Name> during the
QSciTech-QuantumBC virtual workshop on gate-based quantum computing.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from pauli_string import PauliString, LinearCombinaisonPauliString
import numpy as np


class Mapping(object):

    def fermionic_hamiltonian_to_linear_combinaison_pauli_string(self, fermionic_hamiltonian):
        """
        Do the mapping of a FermionicHamiltonian. First generates the LCPS representation of the creation/annihilation
        operators for the specific mapping. Uses the 'to_linear_combinaison_pauli_string' of the FermionicHamiltonian
        to generate the complete LCPS.

        Args:
            fermionic_hamiltonian (FermionicHamiltonian): A FermionicHamiltonian that provided a 
                'to_linear_combinaison_pauli_string' method.

        Returns:
            LinearCombinaisonPauliString: The LCPS reprenseting the FermionicHamiltonian
        """

        aps, ams = self.fermionic_operator_linear_combinaison_pauli_string(fermionic_hamiltonian.number_of_orbitals())
        pslc = fermionic_hamiltonian.to_linear_combinaison_pauli_string(aps, ams)
        return pslc


class JordanWigner(Mapping):
    def __init__(self):
        """
        The Jordan-Wigner mapping
        """

        self.name = 'jordan-wigner'

    def fermionic_operator_linear_combinaison_pauli_string(self, n_qubits):
        """
        Build the LCPS reprensetations for the creation/annihilation operator for each qubit following 
        Jordan-Wigner mapping.

        Args:
            n_qubits (int): The number of orbitals to be mapped to the same number of qubits.

        Returns:
            list<LinearCombinaisonPauliString>, list<LinearCombinaisonPauliString>: Lists of the creation/annihilation
                operators for each orbital in the form of LinearCombinaisonPauliString.
        """
        
        r_bits = [np.concatenate(([True]*n,[False]*n_qubits,[True],[False]*(n_qubits-1-n))) for n in range(n_qubits)]
        i_bits = [np.concatenate(([True]*(n+1),[False]*(n_qubits-1),[True],[False]*(n_qubits-1-n))) for n in range(n_qubits)]
        
        aps = [0.5*PauliString.from_zx_bits(r)-0.5j*PauliString.from_zx_bits(i) for r,i in zip(r_bits,i_bits)]
        ams = [0.5*PauliString.from_zx_bits(r)+0.5j*PauliString.from_zx_bits(i) for r,i in zip(r_bits,i_bits)]
        
        return aps, ams


class Parity(Mapping):
    def __init__(self):
        """
        The Parity mapping
        """

        self.name = 'parity'

    def fermionic_operator_linear_combinaison_pauli_string(self, n_qubits):
        """
        Build the LCPS reprensetations for the creation/annihilation operator for each qubit following 
        Parity mapping.

        Args:
            n_qubits (int): The number of orbtials to be mapped to the same number of qubits.

        Returns:
            list<LinearCombinaisonPauliString>, list<LinearCombinaisonPauliString>: Lists of the creation/annihilation
                operators for each orbital in the form of LinearCombinaisonPauliString
        """

        r0_bits = [np.concatenate(([False]*n_qubits,[True]*n_qubits))]
        r_bits = r0_bits + [np.concatenate(([False]*(n-1),[True],[False]*n_qubits,[True]*(n_qubits-n))) for n in range(1,n_qubits)]
        i_bits = [np.concatenate(([False]*n, [True],[False]*(n_qubits-1),[True]*(n_qubits-n))) for n in range(0,n_qubits)]
        
        aps = [0.5*PauliString.from_zx_bits(r)-0.5j*PauliString.from_zx_bits(i) for r,i in zip(r_bits,i_bits)]
        ams = [0.5*PauliString.from_zx_bits(r)+0.5j*PauliString.from_zx_bits(i) for r,i in zip(r_bits,i_bits)]
        
        return aps, ams
