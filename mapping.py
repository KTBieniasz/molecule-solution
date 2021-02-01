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
        operators for the specific mapping. Uses the 'to_pauli_string_linear_combinaison' of the FermionicHamiltonian
        to generate the complete LCPS.

        Args:
            fermionic_hamiltonian (FermionicHamiltonian): A FermionicHamiltonian that provided a 
                'to_pauli_string_linear_combinaison' method.

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

        aps = [0.5*PauliString.from_str("IIIX")-0.5j*PauliString.from_str("IIIY"),
               0.5*PauliString.from_str("IIXZ")-0.5j*PauliString.from_str("IIYZ"),
               0.5*PauliString.from_str("IXZZ")-0.5j*PauliString.from_str("IYZZ"),
               0.5*PauliString.from_str("XZZZ")-0.5j*PauliString.from_str("YZZZ")]
        ams = [0.5*PauliString.from_str("IIIX")+0.5j*PauliString.from_str("IIIY"),
               0.5*PauliString.from_str("IIXZ")+0.5j*PauliString.from_str("IIYZ"),
               0.5*PauliString.from_str("IXZZ")+0.5j*PauliString.from_str("IYZZ"),
               0.5*PauliString.from_str("XZZZ")+0.5j*PauliString.from_str("YZZZ")]
        
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

        aps = list()
        ams = list()
        
        ################################################################################################################
        # YOUR CODE HERE
        # OPTIONAL
        ################################################################################################################

        raise NotImplementedError()

        return aps, ams
