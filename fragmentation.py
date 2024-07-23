"""
This provides a class `Fragmentation` that simulates the fragmentation process of a molecule given a SMILES string,
an energy budget and compares it to a reference spectrum. It includes functionalities for building a fragmentation tree, plotting
normalized spectra, visualizing the molecule, and plotting the fragmentation tree.

Classes:
    - Fragmentation: A class that models the fragmentation process of a molecule.

Usage example:
    from fragmentation import Fragmentation
    smile = "CCO"
    reference_spectrum = zip([10, 20, 30], [1, 2, 3])
    energy_budget = 40.0
    frag = Fragmentation(smile, reference_spectrum, energy_budget)
    frag.plotNormalizedSpectrum()
    frag.plotMolecule()
"""

from io import StringIO
import numpy as np
import networkx as nx
import pandas as pd
from pysmiles import read_smiles
import matplotlib.pyplot as plt

class Fragmentation:
    """
    A class to model the fragmentation process of a molecule.

    Attributes:
        __smile (str): SMILES string representation of the molecule.
        __reference_spectrum (zip): Reference mass spectrum.
        __energyBuget (float): Available energy budget for fragmentation.
        __bond_energy_map (pd.DataFrame): Bond energy map.
        __molecule (nx.Graph): Graph representation of the molecule.
        __molecule_woH (nx.Graph): Molecule graph without hydrogen atoms.
        __fragmentTree (nx.DiGraph): Fragmentation tree.

    Methods:
        plotNormalizedSpectrum(): Plots the normalized mass spectrometry spectrum.
        plotMolecule(): Plots the molecule structure.
        plotPngFragmentTree(): Plots the fragmentation tree as a PNG image.
        plotHtmlFragmentTree(): Plots the fragmentation tree as an interactive HTML.
    """

    def __init__(self, smile: str, reference_spectrum: zip, energyBuget: float) -> None:
        """
        Initializes the Fragmentation class with a SMILES string, reference spectrum, and energy budget.

        Args:
            smile (str): SMILES string representation of the molecule.
            reference_spectrum (zip): Reference mass spectrum.
            energyBuget (float): Available energy budget for fragmentation.
        """
        
        # constructor -> instance variables
        self.__smile = smile
        self.__reference_spectrum = reference_spectrum
        self.__energyBuget = energyBuget

        # definitions
        self.__kb = 1.380649e-23 # J / K
        self.__n_A =  6.02214076e23 # 1 / mol
        self.__eV_J = 1.6022e-19 # J / eV
        self.__eV_J_mol = 96_485 # J / mol eV
        self.__E_kin = 40.0 # eV

        self.__bond_energy_map = pd.read_json(
                StringIO(
                        '{"energy":{"0":4.4773799036,"1":5.8558325128,"2":4.4255583769,"3":3.7622428357,"4":3.0574700731,"5":4.2804581023,"6":3.5964139504,"7":3.1611131264,"8":3.710421309,"9":5.0266880862,"10":3.5134995077,"11":2.8605482718,"12":2.4874332798,"13":2.6843550811,"14":4.052443385,"15":1.6582888532,"16":2.8190910504,"17":2.0728610665,"18":2.5185261958,"19":2.0832253718,"20":4.8401305902,"21":1.5131885785,"22":1.9692180132,"23":2.1039539825,"24":2.4252474478,"25":1.5961030212,"26":2.6221692491,"27":2.4563403638,"28":2.4770689745,"29":2.2594185625,"30":2.0003109292,"31":1.5442814945,"32":2.1557755091,"33":1.8137534332,"34":3.5964139504,"35":3.3891278437,"36":2.6221692491,"37":2.2594185625,"38":2.7569052184,"39":3.523863813,"40":4.0731719956,"41":3.7311499197,"42":4.6846660103,"43":6.3636834741,"44":8.6956521739,"45":5.1303311396,"46":7.7214074727,"47":11.1105353164,"48":6.2911333368,"49":4.332279629,"50":9.7528113178,"51":9.2345960512,"52":6.3740477794},"order":{"0":1,"1":1,"2":1,"3":1,"4":1,"5":1,"6":1,"7":1,"8":1,"9":1,"10":1,"11":1,"12":1,"13":1,"14":1,"15":1,"16":1,"17":1,"18":1,"19":1,"20":1,"21":1,"22":1,"23":1,"24":1,"25":1,"26":1,"27":1,"28":1,"29":1,"30":1,"31":1,"32":1,"33":1,"34":1,"35":1,"36":1,"37":1,"38":1,"39":1,"40":1,"41":1,"42":1,"43":2,"44":3,"45":2,"46":2,"47":3,"48":2,"49":2,"50":3,"51":3,"52":2},"side_a":{"0":"H","1":"H","2":"H","3":"H","4":"H","5":"C","6":"C","7":"C","8":"C","9":"C","10":"C","11":"C","12":"C","13":"C","14":"N","15":"N","16":"N","17":"N","18":"N","19":"N","20":"O","21":"O","22":"O","23":"O","24":"O","25":"F","26":"F","27":"F","28":"Cl","29":"Cl","30":"Br","31":"I","32":"I","33":"I","34":"S","35":"S","36":"S","37":"S","38":"S","39":"Si","40":"Si","41":"Si","42":"Si","43":"C","44":"C","45":"O","46":"C","47":"C","48":"N","49":"N","50":"N","51":"C","52":"C"},"side_b":{"0":"H","1":"F","2":"Cl","3":"Br","4":"I","5":"H","6":"C","7":"N","8":"O","9":"F","10":"Cl","11":"Br","12":"I","13":"S","14":"H","15":"N","16":"F","17":"Cl","18":"Br","19":"O","20":"H","21":"O","22":"F","23":"Cl","24":"I","25":"F","26":"Cl","27":"Br","28":"Cl","29":"Br","30":"Br","31":"I","32":"Cl","33":"Br","34":"H","35":"F","36":"Cl","37":"Br","38":"S","39":"Si","40":"H","41":"C","42":"O","43":"C","44":"C","45":"O","46":"O","47":"O","48":"O","49":"N","50":"N","51":"N","52":"N"}}'
                    )
                )
        self.__bond_energy_map
        self.__atomic_numbers = [
            ("H" ,1), ("HE", 2), ("LI", 3), ("BE", 4), ("B" ,5), ("C" ,6), ("N" ,7), ("O" ,8), ("F" ,9), ("NE", 10), ("NA", 11), ("MG", 12), ("AL", 13), ("SI", 14), ("P", 15), ("S", 16), ("CL", 17), ("AR", 18), ("K", 19), ("CA", 20), ("SC", 21), ("TI", 22), ("V" ,23), ("CR", 24), ("MN", 25), ("FE", 26), ("NI", 27), ("CO", 28), ("CU", 29), ("ZN", 30), ("GA", 31), ("GE", 32), ("AS", 33), ("SE", 34), ("BR", 35), ("KR", 36), ("RB", 37), ("SR", 38), ("Y" ,39), ("ZR", 40), ("NB", 41), ("MO", 42), ("TC", 43), ("RU", 44), ("RH", 45), ("PD", 46), ("AG", 47), ("CD", 48), ("IN", 49), ("SN", 50), ("SB", 51), ("TE", 52), ("I" ,53), ("XE", 54), ("CS", 55), ("BA", 56), ("LA", 57), ("CE", 58), ("PR", 59), ("ND", 60), ("PM", 61), ("SM", 62), ("EU", 63), ("GD", 64), ("TB", 65), ("DY", 66), ("HO", 67), ("ER", 68), ("TM", 69), ("YB", 70), ("LU", 71), ("HF", 72), ("TA", 73), ("W" ,74), ("RE", 75), ("OS", 76), ("IR", 77), ("PT", 78), ("AU", 79), ("HG", 80), ("TL", 81), ("PB", 82), ("BI", 83), ("TH", 90), ("PA", 91), ("U" ,92), ("NP", 93), ("PU", 94), ("AM", 95), ("CM", 96), ("BK", 97), ("CF", 98), ("ES", 99), ("FM", 100), ("MD", 101), ("NO", 102), ("LR", 103), ("RF", 104), ("DB", 105), ("SG", 106), ("BH", 107), ("HS", 108), ("MT", 109), ("DS", 110), ("RG", 111), ("CN", 112), ("NH", 113), ("FL", 114), ("MC", 115), ("LV", 116), ("TS", 117), ("OG", 118)
        ]
        self.__monoisotopic_masses = [
            0.00000000000, # padding 
            1.00782503223, 3.0160293201, 6.0151228874, 9.012183065, 10.01293695, 12.0000000, 14.00307400443, 15.99491461957, 18.99840316273, 19.9924401762, 22.989769282, 23.985041697, 26.98153853, 27.97692653465, 30.97376199842, 31.9720711744, 34.968852682, 35.967545105, 38.9637064864, 39.962590863, 44.95590828, 45.95262772, 49.94715601, 49.94604183, 54.93804391, 53.93960899, 58.93319429, 57.93534241, 62.92959772, 63.92914201, 68.9255735, 69.92424875, 74.92159457, 73.922475934, 78.9183376, 77.92036494, 84.9117897379, 83.9134191, 88.9058403, 89.9046977, 92.906373, 91.90680796, 96.9063667, 95.90759025, 102.905498, 101.9056022, 106.9050916, 105.9064599, 112.90406184, 111.90482387, 120.903812, 119.9040593, 126.9044719, 123.905892, 132.905451961, 129.9063207, 137.9071149, 135.90712921, 140.9076576, 141.907729, 144.9127559, 143.9120065, 150.9198578, 151.9197995, 158.9253547, 155.9242847, 164.9303288, 161.9287884, 168.9342179, 167.9338896, 174.9407752, 173.9400461, 179.9474648, 179.9467108, 184.9529545, 183.9524885, 190.9605893, 189.9599297, 196.96656879, 195.9658326, 202.9723446, 203.973044, 208.9803991, 208.9824308, 209.9871479, 210.9906011, 223.019736, 223.0185023, 227.0277523, 230.0331341, 231.0358842, 233.0396355, 236.04657, 238.0495601, 241.0568293, 243.0613893, 247.0703073, 249.0748539, 252.08298, 257.0951061, 258.0984315, 259.10103, 262.10961, 267.12179, 268.12567, 271.13393, 272.13826, 270.13429, 276.15159, 281.16451, 280.16514, 285.17712, 284.17873, 289.19042, 288.19274, 293.20449, 292.20746, 294.21392
        ]

        # build molecule graph from SMILE String
        # zero_order_bonds=True needs to be True otherwiese Molecule Graph is not connnected
        #self.__molecule = read_smiles(self.__smile, explicit_hydrogen=True, zero_order_bonds=True, reinterpret_aromatic=True)
        self.__molecule = read_smiles(self.__smile, explicit_hydrogen=True, zero_order_bonds=True, reinterpret_aromatic=True)#, strict=True)

        self.__molecule_woH = self.__molecule
        elements = nx.get_node_attributes(self.__molecule, name = "element").items()
        toRemove = list(node for node, attr in elements if attr == 'H')
        self.__molecule_woH.remove_nodes_from(toRemove)

        self.__addSimpleEnergies()
        
        self.__addBolzmannProbabilityFromEnergy()
        
        # fragmentTree 
        self.__fragmentTree = nx.DiGraph()
        self.__fragmentTree.add_node(0)
        self.__fragmentTree.nodes[0]['fragmentMoleculeIds'] = self.__molecule.nodes()

        self.__build_fragmentation_tree(current_node_id=0, probability=1, energyBudgetRemainung=self.__energyBuget)

        self.predictedSpectrum = self.__fragmentTree2spectralDict()
    
    def plotNormalizedSpectrum(self) -> None:
        """
        Plots the normalized mass spectrometry spectrum, comparing the predicted spectrum with the reference spectrum.
        """
        _, intensities = zip(*self.predictedSpectrum) # Unzip the data into two lists: masses and intensities
        # Normalize the intensities
        max_intensity = max(intensities)
        normalized_A = [(mass, intensity / max_intensity) for mass, intensity in self.predictedSpectrum]
        massesA, intensitiesA = zip(*normalized_A)

        _, intensities = zip(*self.__reference_spectrum) # Unzip the data into two lists: masses and intensities
        # Normalize the intensities
        max_intensity = max(intensities)
        normalized_B = [(mass, intensity / max_intensity) for mass, intensity in self.__reference_spectrum]
        massesB, intensitiesB = zip(*normalized_B)

        # Create the plot
        plt.figure(figsize=(10, 6))
        # blue color
        plt.stem(massesA, intensitiesA, linefmt='b-', markerfmt='bo', basefmt='k', label='Prediction')
        if self.__reference_spectrum != None:
            # Plot dataset with red color
            plt.stem(massesB, intensitiesB, linefmt='r-', markerfmt='ro', basefmt='k', label='Reference Spectrum')

        plt.xlabel('m/z')
        plt.ylabel('Probability')
        plt.title('Mass Spectrometry Spectrum')
        plt.grid(False)
        plt.legend()
        plt.show()
    
    def plotMolecule(self) -> None:
        """
        Plots the molecule structure using NetworkX.
        """
        import matplotlib.pyplot as plt
        elements = nx.get_node_attributes(self.__molecule, name = "element")
        nx.draw(self.__molecule, with_labels=True, labels = elements, pos=nx.spring_layout(self.__molecule))
        plt.gca().set_aspect('equal')
        plt.show()

    def plotPngFragmentTree(self) -> None:
        """
        Plots the fragmentation tree as a PNG image using Matplotlib and NetworkX.
        """
        # Draw the graph
        pos = nx.kamada_kawai_layout(self.__fragmentTree)
        #pos = nx.planar_layout(fragmentTree)
        pos = nx.spring_layout(self.__fragmentTree)
        nx.draw(self.__fragmentTree, pos, with_labels=True)
        plt.show()

    def plotHtmlFragmentTree(self) -> None:
        """
        Plots the fragmentation tree as an interactive HTML using PyVis.
        """
        # https://ona-book.org/gitbook/viz-graphs.html#interactive-visualization-using-networkx-and-pyvis
        from pyvis.network import Network
        from IPython.display import display, HTML

        # create pyvis Network object
        net = Network(notebook = True, cdn_resources='in_line') #height = "500px", width = "600px",

        # as pyvis can only handle certain attributes -> remove all attrib.
        plotGraph = self.__fragmentTree.copy()
        attribute_to_remove = 'fragmentGraph'
        for node in plotGraph.nodes:
            if attribute_to_remove in plotGraph.nodes[node]:
                del plotGraph.nodes[node][attribute_to_remove]


        net.from_nx(plotGraph)
        #net.show('out1.html') # not working in vs code
        #net.save_graph("networkx-pyvis.html")
        HTML(filename="networkx-pyvis.html") # badly working in vs code

    def __build_fragmentation_tree(self, current_node_id: int, probability: float, energyBudgetRemainung: float) -> float:
        """
        Builds the fragmentation tree recursively by partitioning the molecule.

        Args:
            current_node_id (int): The current node ID in the fragmentation tree.
            probability (float): The current probability of the fragmentation.
            energyBudgetRemainung (float): The remaining energy budget.

        Returns:
            float: The probability of the fragment cut.
        """
        fragmentMolecule = self.__molecule.subgraph(self.__fragmentTree.nodes[current_node_id]['fragmentMoleculeIds'])
        if fragmentMolecule.number_of_nodes() < 2:
            return  0 # Base case: reached the maximum depth
        else:
            left_child_id = current_node_id * 2 + 1  # calculate left child id
            right_child_id = current_node_id * 2 + 2  # calculate right child id

            # Add edges from the current node to its children
            self.__fragmentTree.add_edge(current_node_id, left_child_id) # The nodes will be automatically added if they are not already in the graph.
            self.__fragmentTree.add_edge(current_node_id, right_child_id)

            cut_value, partition = nx.stoer_wagner(fragmentMolecule, weight='boltzmannProbability')
            bondEnergy = np.log(cut_value) * (-2/3) * self.__E_kin
            fragmentCutProbability = cut_value

            cut_value, partition = nx.stoer_wagner(fragmentMolecule, weight='bondEnergy')
            bondEnergy = cut_value
            fragmentCutProbability = np.exp(-(3/2) * bondEnergy / self.__E_kin)

            print(bondEnergy, energyBudgetRemainung)

            if bondEnergy > energyBudgetRemainung:
                print("Energy budget used up!")
                return 0 # no more energy remaining

            # Sorted B-Tree: * smaller fragments left; * bigger fragments right
            if self.__molecule.subgraph(partition[0]).number_of_edges() < self.__molecule.subgraph(partition[0]).number_of_edges():
                self.__fragmentTree.nodes[left_child_id]['fragmentMoleculeIds'] = partition[0]
                self.__fragmentTree.nodes[right_child_id]['fragmentMoleculeIds'] = partition[1]
                
            else:
                self.__fragmentTree.nodes[left_child_id]['fragmentMoleculeIds'] = partition[1]
                self.__fragmentTree.nodes[right_child_id]['fragmentMoleculeIds'] = partition[0]
            
            self.__fragmentTree.nodes[left_child_id]['fragmentMass'] = \
                self.__fragment2mass(self.__molecule.subgraph(self.__fragmentTree.nodes[right_child_id]['fragmentMoleculeIds']))
            self.__fragmentTree.nodes[right_child_id]['fragmentMass'] = \
                self.__fragment2mass(self.__molecule.subgraph(self.__fragmentTree.nodes[right_child_id]['fragmentMoleculeIds']))

            self.__fragmentTree.nodes[left_child_id]['fragmentProbability'] = fragmentCutProbability + probability
            self.__fragmentTree.nodes[right_child_id]['fragmentProbability'] = fragmentCutProbability + probability

            fragmentProbabilityLeft = self.__build_fragmentation_tree(left_child_id, fragmentCutProbability + probability, energyBudgetRemainung - bondEnergy)
            fragmentProbabilityRight  = self.__build_fragmentation_tree(right_child_id, fragmentCutProbability + probability, energyBudgetRemainung - bondEnergy)
            
            # graph.nodes[left_child_id]['fragmentProbability'] = fragmentProbabilityLeft
            # graph.nodes[right_child_id]['fragmentProbability'] = fragmentProbabilityRight

            return fragmentCutProbability + fragmentProbabilityLeft + fragmentProbabilityRight # alle Teilsegemente werden aufsummiert, desshalb haben gorße fragmente hohe Warscheinlichkeit: müsste ich dann nicht die Segmente von Oben herab aufsummieren und nicht von unten nach oben.

    def __addSimpleEnergies(self) -> None:
        """
        Adds simple bond energies to the molecule edges based on predefined bond energy map.
        """
        for u,v,_ in self.__molecule.edges(data=True):
            order_eq = (self.__bond_energy_map['order'] == self.__molecule[u][v]['order'])
            side_a_u = (self.__bond_energy_map['side_a'] == self.__molecule.nodes(data='element')[u])
            side_b_v = (self.__bond_energy_map['side_b'] == self.__molecule.nodes(data='element')[v])
            side_b_u = (self.__bond_energy_map['side_b'] == self.__molecule.nodes(data='element')[u])
            side_a_v = (self.__bond_energy_map['side_a'] == self.__molecule.nodes(data='element')[v])

            df = self.__bond_energy_map.loc[ order_eq & ((side_a_u & side_b_v) | (side_a_v & side_b_u)) ]
            try:
                simple_bond_energy = float(df['energy'].iloc[0]) if 'energy' in  df.columns else np.nan # energy in eV / Bond
            except:
                print("assuming default energy; Bond not found in lookup-tabel")
                simple_bond_energy = 5.0 # eV
            self.__molecule[u][v]['bondEnergy'] = simple_bond_energy

    def __addBolzmannProbabilityFromEnergy(self) -> None:
        """
        Adds Boltzmann probabilities to the molecule edges based on bond energies and kinetic energy.
        """
        bolz = list()

        for u,v in self.__molecule.edges(data=False):
            bondEnergy = self.__molecule[u][v]['bondEnergy'] # eV / Bond
            exponent = -(3/2) * bondEnergy / self.__E_kin # without unit
            self.__molecule[u][v]['boltzmannProbability'] = np.exp(exponent)
            bolz.append(self.__molecule[u][v]['boltzmannProbability'])

        for u,v in self.__molecule.edges(data=False):
            self.__molecule[u][v]['boltzmannProbability'] = (self.__molecule[u][v]['boltzmannProbability'] / sum(bolz))
        
    def __fragmentTree2spectralDict(self) -> list:
        """
        Converts the fragmentation tree to a spectral dictionary.

        Returns:
            list: A list of tuples representing the mass and probability of each fragment.
        """
        fragmentMasses = nx.get_node_attributes(self.__fragmentTree, name ='fragmentMass')
        fragmentProbabilities = nx.get_node_attributes(self.__fragmentTree, name ='fragmentProbability')
        z1 = list(fragmentMasses.values())
        z2 = list(fragmentProbabilities.values())
        return list(zip(z1, z2))
    
    def __fragment2mass(self, sub_mol) -> float:
        """
        Calculates the mass of a sub-molecule.

        Args:
            sub_mol (nx.Graph): Subgraph of the molecule.

        Returns:
            float: Mass of the sub-molecule.
        """
        elements = nx.get_node_attributes(sub_mol, name = "element")
        mass = 0
        for element in elements:
            mass += self.__symbol2mass(sub_mol.nodes(data='element')[element])
        return mass

    def __symbol2mass(self, symbol) -> float:
        """
        Converts a chemical symbol to its corresponding monoisotopic mass.

        Args:
            symbol (str): Chemical symbol.

        Returns:
            float: Monoisotopic mass of the element.

        Raises:
            NameError: If the symbol is not a valid atomic symbol.
        """
        aNbr = [i for sym, i in self.__atomic_numbers if sym == symbol.upper()]
        if len(aNbr) != 1:
            raise(NameError('"' + str(symbol) + '"' + " not a valid Atomic symbol, at least for me"))
        return self.__monoisotopic_masses[aNbr[0]]
