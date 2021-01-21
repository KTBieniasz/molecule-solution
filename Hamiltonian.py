'''
Hamiltonian.py - Define Hamiltonian

(C) 2020-2021 Maxime Dion <maxime.dion@usherbrooke.ca>
'''

import numpy as np
from PauliString import PauliString, LinearCombinaisonPauliString

class FermionicHamiltonian(object):

    def __str__(self):
        """String reprensetation of FermionicHamiltonian

        Returns:
            str: Description of FermionicHamiltonian
        """        

        out = f'Fermionic Hamiltonian'
        out += f'\nNumber of orbitals : {self.number_of_orbitals():d}'
        out += f'\nIncluding spin : {str(self.with_spin)}'
        return out

    def number_of_orbitals(self):
        """Number of orbital in the state basis

        Returns:
            int
        """        

        return self.integrals.shape[0]

    def include_spin(self,order='group_spin'):
        """Transforms a spinless FermionicHamiltonian to inlude spin. 
        Doubles the number of orbitals in the basis following the input order.
        Does nothing if the spin is already included (with_spin is True)

        Args:
            order (str, optional): Controls the order of the basis state. Defaults to 'group_spin'.
                With order as 'group_orbital', the integrals will alternate between spin up and down (g_up,g_down,...)
                With order as 'group_spin', the integrals will gather same spin together (g_up,...,g_down,...)

        Raises:
            ValueError: [description]

        Returns:
            FermionicHamiltonian: Including the spin
        """        

        if self.with_spin:
            print('already with spin')
            return self

        if order == 'group_spin':
            new_integrals = np.kron(self.spin_tensor,self.integrals)
        elif order == 'group_orbital':
            new_integrals = np.kron(self.integrals,self.spin_tensor)
        else:
            raise ValueError("Order should be 'group_spin' or 'group_orbital'.")
        
        return self.__class__(new_integrals,with_spin = True)

    def get_integrals(self,cut_zeros = True,threshold = 1e-9):
        """Return the integral tensor with an optionnal threshold for values close to 0.

        Args:
            cut_zeros (bool, optional): If True, all integral values small than threshold we be put to 0. Defaults to True.
            threshold (float, optional): Value of the threshold. Defaults to 1e-9.

        Returns:
            np.ndarray: The integral tensor
        """        

        integrals = self.integrals.copy()
        integrals[np.abs(integrals) < threshold] = 0
        return integrals


class OneBodyFermionicHamiltonian(FermionicHamiltonian):
    spin_tensor = np.eye(2)
    def __init__(self,integrals,with_spin = False):
        """A FermionicHamiltonian representing a one body term in the form of $sum_i h_{ij} a_i^\dagger a_j$

        Args:
            integrals (np.ndarray): Square tensor (n*n) containing the integral values.
            with_spin (bool, optional): Does the integral tensor includes the spin? Defaults to False.
                Should be False if the integrals is for orbital part only
                Should be True if the spin is already included in the integrals

        Raises:
            ValueError: [description]
        """        

        if not(integrals.ndim == 2):
            raise ValueError('Integral tensor should be ndim == 2 for a one-body hamiltonian')

        self.integrals = integrals
        self.with_spin = with_spin

    def change_basis(self,transform):
        """Transforms the integrals tensor (n*n) into a new basis

        Args:
            transform (np.ndarray): Square tensor (n*n) defining the basis change

        Returns:
            OneBodyFermionicHamiltonian: Transformed Hamiltonian 
        """        

        # TO COMPLETE (after activity 2.2)
        # Hint : make use of np.einsum
        # new_integrals = 
        NotImplementedError()
        return OneBodyFermionicHamiltonian(new_integrals,self.with_spin)

    def to_linear_combinaison_pauli_string(self,aps,ams):
        """Generate a qubit operator reprensentation (LCPS) of the OneBodyFermionicHamiltonian given some creation/annihilation operators

        Args:
            aps (list<LinearCombinaisonPauliString>): List of the creation operator for each orbital in the form of LinearCombinaisonPauliString
            ams (list<LinearCombinaisonPauliString>): List of the annihilation operator for each orbital in the form of LinearCombinaisonPauliString

        Returns:
            LinearCombinaisonPauliString: Qubit operator reprensentation of the OneBodyFermionicHamiltonian
        """        

        n_orbs = self.number_of_orbitals()
        # Since each creation/annihilation operator consist of 2 PauliString for each orbital
        # and we compute ap * am there will be (2*n_orbs)**2 Coefs and PauliStrings
        new_coefs = np.zeros(((2*n_orbs)**2,),dtype = np.complex)
        new_pauli_strings = np.zeros(((2*n_orbs)**2,),dtype = PauliString)
        # TO COMPLETE (after activity 3.1)
        NotImplementedError()
        return lcps


class TwoBodyFermionicHamiltonian(FermionicHamiltonian):
    spin_tensor = np.kron(np.eye(2)[:,None,None,:],np.eye(2)[None,:,:,None]) #physicist notation
    def __init__(self,integrals,with_spin = False):
        """A FermionicHamiltonian representing a two body term in the form of $sum_i h_{ijkl} a_i^\dagger a_j^\dagger a_k a_l$

        Args:
            integrals (np.ndarray): Square tensor (n*n) containing the integral values.
            with_spin (bool, optional): Does the integral tensor includes the spin? Defaults to False.
                Should be False if the integrals is for orbital part only
                Should be True if the spin is already included in the integrals

        Raises:
            ValueError: [description]
        """  

        if not(integrals.ndim == 4):
            raise ValueError('Integral tensor should be ndim == 4 for a two-body hamiltonian')
            
        self.integrals = integrals
        self.with_spin = with_spin

    def change_basis(self,transform):
        """Transforms the integrals tensor (n*n*n*n) into a new basis

        Args:
            transform (np.ndarray): Square tensor (n*n) defining the basis change

        Returns:
            TwoBodyFermionicHamiltonian: Transformed Hamiltonian 
        """    

        # TO COMPLETE (after activity 2.2)
        # Hint : make use of np.einsum
        # new_integrals = 
        NotImplementedError()
        return TwoBodyFermionicHamiltonian(new_integrals,self.with_spin)

    def to_linear_combinaison_pauli_string(self,aps,ams):
        """Generate a qubit operator reprensentation (LCPS) of the TwoBodyFermionicHamiltonian given some creation/annihilation operators

        Args:
            aps (list<LinearCombinaisonPauliString>): List of the creation operator for each orbital in the form of LinearCombinaisonPauliString
            ams (list<LinearCombinaisonPauliString>): List of the annihilation operator for each orbital in the form of LinearCombinaisonPauliString

        Returns:
            LinearCombinaisonPauliString: Qubit operator reprensentation of the TwoBodyFermionicHamiltonian
        """     

        n_orbs = self.number_of_orbitals()
        # Since each creation/annihilation operator consist of 2 PauliString for each orbital
        # and we compute ap * ap * am * am there will be (2*n_orbs)**4 Coefs and PauliStrings
        new_coefs = np.zeros(((2*n_orbs)**4 ,),dtype = np.complex)
        new_pauli_strings = np.zeros(((2*n_orbs)**4,),dtype = PauliString)
        ## TO COMPLETE (after activity 3.1)
        NotImplementedError()
        return lcps
        

class MolecularFermionicHamiltonian(FermionicHamiltonian):
    def __init__(self,one_body,two_body,with_spin = False):
        """A composite FermionicHamiltonian made of 1 OneBodyFermionicHamiltonian and 1 TwoBodyFermionicHamiltonian

        Args:
            one_body (OneBodyFermionicHamiltonian): [description]
            two_body (TwoBodyFermionicHamiltonian): [description]
            with_spin (bool, optional): [description]. Defaults to False.
        """
        if one_body.number_of_orbitals() != two_body.number_of_orbitals():
            raise()

        self.one_body = one_body
        self.two_body = two_body
        self.with_spin = with_spin
    
    @classmethod
    def from_pyscf_mol(cls,mol):
        """Generates a MolecularFermionicHamiltonian describing a Molecule from a pyscf Molecule reprensetation.

        Args:
            mol (pyscf.gto.mole.Mole): Molecule object used to compute different integrals

        Returns:
            MolecularFermionicHamiltonian: The Hamiltonian decribing the Molecule including 1 OneBody and 1 TwoBody
        """

        # TO COMPLETE (after activity 2.3)
        # Hint : Make sure the 2 body integrals are in the physicist notation (order) or change the spin_tensor accordingly
        
        # Diagonalisation of ovlp and build a transformation toward an orthonormal basis (ao2oo)
        # TO COMPLETE

        # Build h1 in AO basis and transform it into OO basis
        # TO COMPLETE

        # Find a transformation from OO basis toward MO basis where h1 is diagonal and eigenvalues are in growing order
        # TO COMPLETE

        # Transform h1 and h2 from AO to MO basis
        # TO COMPLETE
        # h1_mo = 
        # h2_mo = 

        # Build the one and two body Hamiltonians
        one_body = OneBodyFermionicHamiltonian(h1_mo)
        two_body = TwoBodyFermionicHamiltonian(h2_mo)

        # Recommended : Make sure that h1_mo is diagonal and that its eigenvalues are sorted in growing order.
        NotImplementedError()
        return cls(one_body,two_body)

    def number_of_orbitals(self):
        """Number of orbital in the state basis

        Returns:
            int
        """ 

        return self.one_body.integrals.shape[0]

    def change_basis(self,transform):
        """Transforms the integrals tensors for both sub Hamiltonian
        See FermionicHamiltonian.change_basis

        Args:
            transform (np.ndarray): Square tensor (n*n) defining the basis change

        Returns:
            MolecularFermionicHamiltonian: Transformed Hamiltonian 
        """

        new_one_body = self.one_body.change_basis(transform)
        new_two_body = self.two_body.change_basis(transform)
        return MolecularFermionicHamiltonian(new_one_body,new_two_body,self.with_spin)

    def include_spin(self):
        """Transforms a spinless FermionicHamiltonian to inlude spin for both sub Hamiltonian
        See FermionicHamiltonian.include_spin

        Args:
            order (str, optional): Controls the order of the basis state. Defaults to 'group_spin'.
                With order as 'group_orbital', the integrals will alternate between spin up and down (g_up,g_down,...)
                With order as 'group_spin', the integrals will gather same spin together (g_up,...,g_down,...)

        Raises:
            ValueError: [description]

        Returns:
            FermionicHamiltonian: Including the spin
        """  

        if self.with_spin:
            print('already with spin')
            return self

        new_one_body = self.one_body.include_spin()
        new_two_body = self.two_body.include_spin()
        return MolecularFermionicHamiltonian(new_one_body,new_two_body,with_spin = True)

    def get_integrals(self,**vargs):
        
        """Return the integral tensors for both sub Hamiltonian with an optionnal threshold for values close to 0.

        Args:
            cut_zeros (bool, optional): If True, all integral values small than threshold we be put to 0. Defaults to True.
            threshold (float, optional): Value of the threshold. Defaults to 1e-9.

        Returns:
            np.ndarray, np.ndarray: The integral tensors
        """ 

        integrals_one = self.one_body.get_integrals(**vargs)
        integrals_two = self.two_body.get_integrals(**vargs)
        return integrals_one, integrals_two

    def to_linear_combinaison_pauli_string(self,aps,ams):
        """Generate a qubit operator reprensentation (LCPS) of the MolecularFermionicHamiltonian given some creation/annihilation operators

        Args:
            aps (list<LinearCombinaisonPauliString>): List of the creation operator for each orbital in the form of LinearCombinaisonPauliString
            ams (list<LinearCombinaisonPauliString>): List of the annihilation operator for each orbital in the form of LinearCombinaisonPauliString

        Returns:
            LinearCombinaisonPauliString: Qubit operator reprensentation of the MolecularFermionicHamiltonian
        """     

        # TO COMPLETE (after activity 3.1)
        NotImplementedError()
        return out

