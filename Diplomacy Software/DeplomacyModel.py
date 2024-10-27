import os
import pickle
from pathlib import Path
from math import exp, log
import random
import country_converter as coco

# Thresholds
risk_threshold = 6
bilateral_update_threshold = 2

# Load the list of countries
with open("Gandalf/FrontEnd/UtilityFiles/countries.bin", "rb") as f:
    """
    Note: 
    When the list NamedCountries is updated, all the weights and node values need to be refreshed. 
    For this purpose, delete ALL the files in the Weights folder.
    """
    NamedCountries = pickle.load(f)
    NamedCountries = coco.convert(names=NamedCountries, to='name_short')

# Define the classes of the nodes
def classes(code):
    """
    This function returns the class of the node based on the code provided.

    code: str: The code of the class. eg. "p1" for political class 1
    """

    high = code[0]
    try:
        low = int(code[1])
    except IndexError:
        low = ""

    if high == "p":
        if low == "":
            return "Political"
        elif low == 1:
            return "War"
        elif low == 2:
            return "Dividedness"
        elif low == 3:
            return "Scandals"
        elif low == 4:
            return "Cultural diplomacy"
        else:
            return "Incorrect code"
    elif high == "e":
        if low == "":
            return "Economic"
        elif low == 1:
            return "Economic instability"
        elif low == 2:
            return "Government policy"
        elif low == 3:
            return "Economic Stagnation"
        else:
            return "Incorrect code"
    else:
        return "Incorrect code"
    
# Create a node: a class that has the behaviour of a node in the graph
class node:
    """
    A class to represent a node in the graph.
    """

    def __init__(self, country, subclass):
        """
        Initialises the node with the country and the subclass of the node.

        country: str: The name of the country
        subclass: str: The subclass of the node
        """

        self.country = country
        self.subclass = subclass

        self.value = 0 # No risk when initialised

        self.events = [] # Store the ids of events that have affected this node
        # self.events this is just for processing and not for data storage; the database performs that function

        self.change = 0 # The change in the risk score

    def UpdateFunction(self, score):
        """
        Updates the value of the node based on the score provided.

        score: float: The score to update the node with
        """

        return - exp(-(score/5 - log(10))) + 10
    
    def update(self, score):
        """
        Updates the value of the node based on the score provided.

        score: float: The score to update the node with
        """

        self.change = score
        score += self.value
        
        self.value = self.UpdateFunction(score)

# Create a class to store bilateral risks between two countries
class bilateral:
    """
    A class to represent the bilateral relations between two countries.
    """

    def __init__(self, countries):
        """
        Initialises the bilateral relations between two countries.

        countries: list: A list of two countries
        """

        self.cause = [] # Refer to an event ID; this is only stored if an event causes a significant change in bilateral relations
        self.countries = [] # Countries is a list of length 2 containing class instances of 2 countries; the country with the smaller ID is ahead

        for country in countries:
            self.countries.append(country.name)

        self.political = node(self.countries, classes("p"))
        self.economic = node(self.countries, classes("e"))

        # Store causes
        self.loc = os.path.join("DeploymentFiles", "Events", "".join(self.countries) + ".bin")
        self.path = Path(self.loc)

        if self.path.is_file():
            with open(self.loc, "rb") as f:
                self.cause = pickle.load(f)
        else:
            with open(self.loc, "wb") as f:
                pickle.dump(self.cause, f)

        """
        As of now this segment is not needed

        1. It needs to be improved to connect to both countries in play
        2. It is more important when we consider layers of propagation
        
        # Weight gives a link from the bilateral relations to the weights of the first country
        self.weights = [[0.5 for i in range(7)] for j in range(2)]

        self.loc = "Weights\\" + "".join(self.countries) + ".bin"
        self.path = Path(self.loc)

        if self.path.is_file():
            with open(self.loc, "rb") as f:
                self.weights = pickle.load(f)
        else:
            self.weights = {"domestic": [[0.5 for i in range(7)] for j in range(2)]}
            with open(self.loc, "wb") as f:
                pickle.dump(self.weights, f)
        """

def GetBilateral(country1id, country2id, countries, BilateralInfo):
    """
    This function returns the bilateral relations between two countries.

    country1id: int: The ID of the first country
    country2id: int: The ID of the second country
    countries: list: A list of class instances of countries
    BilateralInfo: list: A list of class instances of bilateral relations

    Returns: bilateral: The bilateral relations between the two countries
    """

    if country2id < country1id:
        country2id, country1id = country1id, country2id
    a = [i for i in range(len(countries))]

    loopno = 0
    for i in range(len(a)):
        for j in range(len(a) - i - 1):
            if [a[i], a[j + i + 1]] == [country1id, country2id]:
                return BilateralInfo[loopno]
            loopno += 1
            
    raise Exception("This combination of countries is invalid")

def Update(node, change, weight):
    """
    This function updates the value of the node based on the change and the weight provided.

    node: node: The node to update
    change: float: The change in the risk score
    weight: float: The weight of the link

    Returns: float: The update in the risk score
    """

    update = change * weight

    if update > risk_threshold:
        print(f"There is a grave situation in {node.country} in the field of {node.subclass}. The risk increases by the risk score of {update}")
    
    node.update(update)
    return update

# Create a class for the different countries in the graph
class domestic:
    """
    A class to represent a country in the graph.
    """

    def __init__(self, name):
        """
        Initialises the country with the name provided.

        name: str: The name of the country
        """

        self.name = name
        self.id = NamedCountries.index(name)
        self.nodes = []

        # Political
        for i in range(1, 5):
            self.nodes.append(node(self.name, classes("p" + str(i))))

        # Economic
        for i in range(1, 4):
            self.nodes.append(node(self.name, classes("e" + str(i))))

        self.loc = os.path.join("DeploymentFiles", "Weights" + Qual, self.name + ".bin")
        self.path = Path(self.loc)

        if self.path.is_file():
            with open(self.loc, "rb") as f:
                self.weights = pickle.load(f)
        else:
            self.weights = {"domestic": [[random.random() for i in range(7)] for j in range(7)]}
            with open(self.loc, "wb") as f:
                pickle.dump(self.weights, f)

            print(f"The weights linking to {self.name} have been freshly initialised.")

    def AddLink(self, country):
        """
        This function adds a link to the country provided.

        country: class: The class instance of the country to link to
        """

        if self.name == country.name:
            return None

        self.weights[country.name] = [[random.random() for i in range(9)] for j in range(7)]
    
    def FindNode(self, code):
        """
        This function returns the node number in the pre-defined format.

        code: str: The code of the node

        Returns: int: The node number
        """

        if code[0] == "p":
            return int(code[1]) - 1
        elif code[0] == "e":
            return int(code[1]) + 4
        else:
            print("Invalid subclass code")
            return None

    def propagate(self, StartNode, countries, BilateralInfo):
        """
        This function propagates the changes in the risk score of the nodes to the other nodes in the network.

        StartNode: int: The node number to start the propagation from
        countries: list: A list of class instances of countries
        BilateralInfo: list: A list of class instances of bilateral relations
        """

        # Only changes get propagrated, not actual values
        # Self.change has to be updated before this behaviour is called
        # Start node is the number of the node - an integer from 1 to 7

        # Domestic propagation
        for i in range(7):
            if i != StartNode:
                Update(self.nodes[i], self.nodes[StartNode].change, self.weights["domestic"][StartNode][i])
                
                if self.nodes[i].value > 8:
                    print(f"There is a grave situation in {self.name} in the field of {self.nodes[i].subclass} due to an event {self.nodes[StartNode].events[-1]}.")
        
        # International propagation
        for country in self.weights.keys():
            if country != "domestic":
                CountryID = NamedCountries.index(country) # ID of the international country

                for i in range(7):
                    Update(countries[CountryID].nodes[i], self.nodes[StartNode].change, self.weights[country][StartNode][i])
                
                rel = GetBilateral(CountryID, self.id, countries, BilateralInfo)
                i = 7
                if Update(rel.political, self.nodes[StartNode].change, self.weights[country][StartNode][i]) > bilateral_update_threshold:
                    rel.cause.append(self.nodes[StartNode].events[-1])
                
                i = 8
                if Update(rel.economic, self.nodes[StartNode].change, self.weights[country][StartNode][i]) > bilateral_update_threshold:
                    rel.cause.append(self.nodes[StartNode].events[-1])

                if ( rel.political.value + rel.economic.value ) / 2 > 8:
                    print(f"There is a grave situation in the relations between {self.name} and {country} due to the event id {self.nodes[StartNode].events[-1]}.")
                    print(rel.countries, ":", ( rel.political.value + rel.economic.value ) / 2 )

def ExtractNodes(countries, BilateralInfo):
    """
    This function extracts the values of the nodes from the countries and bilateral relations.

    countries: list: A list of class instances of countries
    BilateralInfo: list: A list of class instances of bilateral relations

    Returns: tuple: A tuple of two lists of nodes
    """

    NodeList = []

    for country in countries:
        NodeList += country.nodes

    NodeListIntl = []

    for relation in BilateralInfo:
        NodeListIntl += [relation.political, relation.economic]

    return NodeList, NodeListIntl

def ReloadNodes(NodeList, NodeListIntl, countries, BilateralInfo):
    """
    This function reloads the values of the nodes from the lists provided into the countries and bilateral relations.

    NodeList: list: A list of nodes
    NodeListIntl: list: A list of nodes
    countries: list: A list of class instances of countries
    BilateralInfo: list: A list of class instances of bilateral relations
    """
    
    for country in countries:
        country.nodes = NodeList[0:len(country.nodes)]

        del NodeList[0:len(country.nodes)]

    if NodeList == []:
        print("Successful reloading")
    else:
        raise Exception("Reload error")
    
    i = 0

    for relation in BilateralInfo:
        relation.political = NodeListIntl[i + 0]
        relation.economic = NodeListIntl[i + 1]
        i += 2

def ReadNodes(countries, BilateralInfo):
    """
    This function reads the values of the nodes from the storage.

    countries: list: A list of class instances of countries
    BilateralInfo: list: A list of class instances of bilateral relations
    """

    loc = os.path.join("DeploymentFiles", "Weights" + Qual, "Nodes.bin")
    try:
        with open(loc, "rb") as f:
            nodes = pickle.load(f)
        ReloadNodes(nodes[0], nodes[1], countries, BilateralInfo)
        print("Nodes have been successfully loaded.")
    except FileNotFoundError:
        AllNodes = ExtractNodes(countries, BilateralInfo)
        with open(loc, "wb") as f:
            pickle.dump(AllNodes, f)
        print("Nodes freshly initialised in Nodes.bin")

def SaveNodes(countries, BilateralInfo):
    """
    This function saves the values of the nodes to the storage.

    countries: list: A list of class instances of countries
    BilateralInfo: list: A list of class instances of bilateral relations
    """

    AllNodes = ExtractNodes(countries, BilateralInfo)
    loc = os.path.join("Weights", Qual, "Nodes.bin")
    with open(loc, "wb") as f:
        pickle.dump(AllNodes, f)
    print("Nodes saved in Nodes.bin")  

def InitModel(Q):
    """
    This function initialises the model.

    Q: bool: A boolean to determine if the model is a Qualitative model

    Returns: tuple: A tuple of two lists of class instances of countries and bilateral relations
    """

    global Qual
    Qual = "Qual" if Q else ""
    
    # Intialise countries
    countries = []
    for name in NamedCountries:
        countries.append(domestic(name))

    # Intialise bilateral relations
    BilateralInfo = []

    for i in range(len(countries)):
        for j in range(len(countries) - i - 1):
            BilateralInfo.append(bilateral([countries[i], countries[j + i + 1]]))

    # Initalise links
    for i in range(len(countries)):
        for j in range(len(countries)):
            if i != j:
                countries[i].AddLink(countries[j])

    ReadNodes(countries, BilateralInfo)
    print("The value of nodes has been extracted from storage and loaded into the model.")
    return countries, BilateralInfo

