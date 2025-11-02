from rdflib import Graph, Literal, RDF, URIRef
from rdflib.namespace import FOAF , XSD, RDF, OWL
import pdb
import csv
import os
import xmltodict
import xml.etree.ElementTree as ET
from rdflib import Namespace
import numpy as np
from tensorflow.python.keras.utils import object_identity

def get_refined_classes():
    refined_classes = {
        "Bird" : [ "Animals" ],
        "Cat" : [ "Animals" ],
        "Cow" : [ "Animals" ],
        "Dog" : [ "Animals" ],
        "Horse" : [ "Animals" ],
        "Person" : [ "Animals" ],
        "Sheep" : [ "Animals" ],
        "Aeroplane" : [ "Vehicle" ], 
        "Bicylcle" : [ "Vehicle" ],
        "Boat" : [ "Vehicle" ],
        "Car" : [ "Vehicle" ],
        "Motorbike" : [ "Vehicle" ],
        "Train" : ["Vehicle"]
        }
    return refined_classes


def get_component_classes():
    component_classes = {
        "Animals" : [ "Eye", "Head", "Leg", "Neck", "Torso" ],
        "Bird" : ["Animal_Wing", "Beak", "Tail"],
        "Cat" : ["Ear", "Tail"],
        "Cow" : ["Ear", "Horn", "Muzzle", "Tail"],
        "Dog" : ["Ear", "Muzzle", "Nose", "Tail"],
        "Horse" : ["ear", "Hoof", "Muzzle", "Tail"],
        "Person" : ["Arm", "Ear", "Ebrow", "Foot", "Hair", "Hand", "Mouth", "Nose"],
        "Sheep" : ["Ear", "Horn", "Muzzle", "Tail"],
        "Bottle" : ["Body", "Cap"],
        "Pottedplant" : ["Plant", "Pot"],
        "Tvmonitor" : ["Screen"],
        "Aeroplane" : ["Artifact_Wing", "Body", "Engine", "Stern", "Wheel"],
        "Bicycle" : ["Chain_Wheel", "Handlebar", "Headlight", "Saddle", "Wheel"],
        "Bus" : ["Bodywork", "Door", "Headlight", "License_plate", "Mirror", "Wheel", "Window"],
        "Car" : ["Bodywork", "Door", "Headlight", "License_plate", "Mirror", "Wheel", "Window"],
        "Motorbike" : ["Handlebar", "Headlight", "Saddle", "Wheel"],
        "Train" : [ "Coach", "Headlight", "Locomotive" ]
        }
    return component_classes

# ensemble de classes
# returns : dictionnaire 
def associate_number_to_class_and_class_to_number(classe_names):
    dictionnary_number_to_class = {}
    dictionnary_class_to_number = {}
    indice = 0
    for element in classe_names:
        dictionnary_number_to_class[indice] = element
        dictionnary_class_to_number[element] = indice
        indice += 1
    return dictionnary_number_to_class, dictionnary_class_to_number

def associate_codes_to_hierarchies(codes, named_hierarchy):
    coded_named_hierarchy = {}
    for key in named_hierarchy:
        values = named_hierarchy.get(key)
        coded_values = []
        for element in values:
            element_code = codes.get(element)
            coded_values.append(element_code)
        coded_key = codes.get(key)
        coded_named_hierarchy[coded_key] = coded_values

    return coded_named_hierarchy

def get_class_names():
    classes = {
        0 : "Aeroplane",
        1 : "Animal_Wing",
        2 : "Animal",
        3 : "Arm",
        4 : "Artifact_Wing",
        5 : "Beak",
        6 : "Bicycle",
        7 : "Bird",
        8 : "Boat",
        9 : "Body",
        10 : "Bodywork",
        11 : "Bottle",
        12 : "Bus",
        13 : "Cap",
        14 : "Car",
        15 : "Cat",
        16 : "Chain_Wheel",
        17 : "Chair",
        18 : "Coach",
        19 : "Cow",
        20 : "Dining_Table",
        21 : "Dog",
        22 : "Door",
        23 : "Ear",
        24 : "Eyebrow",
        25 : "Engine",
        26 : "Eye",
        27 : "Foot",
        # num : "Furniture"
        28 : "Hair",
        29 : "Hand",
        30 : "Handlebar",
        31 : "Head",
        32 : "Headlight",
        33 : "Hoof",
        34 : "Horn",
        35 : "Horse",
        36 : "Leg",
        37 : "License_Plate",
        38 : "Locomotive",
        39 : "Mirror",
        40 : "Motorbike",
        41 : "Mouth",
        42 : "Muzzle",
        43 : "Neck",
        44 : "Nose",
        45 : "Person",
        46 : "Plant",
        47 : "Pot",
        48 : "Potted_Plant",
        49 : "Saddle",
        50 : "Screen",
        51 : "Sheep",
        52 : "Sofa",
        53 : "Stern",
        54 : "Tail",
        55 : "Torso",
        56 : "Train",
        57 : "Television_Monitor",
        58 : "Vehicle",
        59 : "Wheel",
        60 : "Window"
    }
    return classes

def get_yolo_class_codes():
    codes = {
        "Aeroplane" : 0,
        "Animal_wing" : 1,
        "Animals" : 2,
        "Arm" : 3,
        "Artifact_wing" : 4,
        "Beak" : 5,
        "Bicycle" : 6,
        "Bird" : 7,
        "Boat" : 8,
        "Body" : 9,
        "Bodywork" : 10,
        "Bottle" : 11,
        "Bus" : 12,
        "Cap" : 13,
        "Car" : 14,
        "Cat" : 15,
        "Chain_wheel" : 16,
        "Chair" : 17,
        "Coach" : 18,
        "Cow" : 19,
        "Diningtable" : 20,
        "Dog" : 21,
        "Door" : 22,
        "Ear" : 23,
        "Ebrow" : 24,
        "Engine" : 25,
        "Eye" : 26,
        "Foot" : 27,
        "Hair" : 28,
        "Hand" : 29,
        "Handlebar" : 30,
        "Head" : 31,
        "Headlight" : 32,
        "Hoof" : 33,
        "Horn" : 34,
        "Horse" : 35,
        "Leg" : 36,
        "License_plate" : 37,
        "Locomotive" : 38,
        "Mirror" : 39,
        "Motorbike" : 40,
        "Mouth" : 41,
        "Muzzle" : 42,
        "Neck" : 43,
        "Nose" : 44,
        "Person" : 45,
        "Plant" : 46,
        "Pot" : 47,
        "Pottedplant" : 48,
        "Saddle" : 49,
        "Screen" : 50,
        "Sheep" : 51,
        "Sofa" : 52,
        "Stern" : 53,
        "Tail" : 54,
        "Torso" : 55,
        "Train" : 56,
        "Tvmonitor" : 57,
        "Vehicle" : 58,
        "Wheel" : 59,
        "Window" : 60
        }
    return codes

def add_refined_classes(refined_classes_codes, object_class):
    km_line = ''
    object_class_refined_classes = refined_classes_codes.get(object_class,[])
    for refined_class in object_class_refined_classes:
        km_line = km_line + ' ' + str(refined_class)
    return km_line

def extract_bb_coordinates(id, ann_dict):
	if id > -1:
		x_1 = int(ann_dict[id]['polygon']['pt'][0]['x'])
		y_1 = int(ann_dict[id]['polygon']['pt'][0]['y'])
		x_2 = int(ann_dict[id]['polygon']['pt'][1]['x'])
		y_2 = int(ann_dict[id]['polygon']['pt'][1]['y'])
		x_3 = int(ann_dict[id]['polygon']['pt'][2]['x'])
		y_3 = int(ann_dict[id]['polygon']['pt'][2]['y'])
		x_4 = int(ann_dict[id]['polygon']['pt'][3]['x'])
		y_4 = int(ann_dict[id]['polygon']['pt'][3]['y'])
	else:
		x_1 = int(ann_dict['polygon']['pt'][0]['x'])
		y_1 = int(ann_dict['polygon']['pt'][0]['y'])
		x_2 = int(ann_dict['polygon']['pt'][1]['x'])
		y_2 = int(ann_dict['polygon']['pt'][1]['y'])
		x_3 = int(ann_dict['polygon']['pt'][2]['x'])
		y_3 = int(ann_dict['polygon']['pt'][2]['y'])
		x_4 = int(ann_dict['polygon']['pt'][3]['x'])
		y_4 = int(ann_dict['polygon']['pt'][3]['y'])
	xmin = np.min([x_1, x_2, x_3, x_4])
	ymin = np.min([y_1, y_2, y_3, y_4])
	xmax = np.max([x_1, x_2, x_3, x_4])
	ymax = np.max([y_1, y_2, y_3, y_4])
	return xmin, ymin, xmax, ymax


class PASCALPart_annotations:

    def __init__(self):
        self.split = ""
        # this is just an annotation example useful for your own "fun"
        self.annotations = {
            "00001": {
                "1": {
                    "class": "Person",
                    "x_1": 1,
                    "y_1": 1,
                    "x_2": 123,
                    "y_2": 124,
                    "isPartOf": ""
                },
                "2": {
                    "class": "Leg",
                    "x_1": 23,
                    "y_1": 23,
                    "x_2": 44,
                    "y_2": 44,
                    "isPartOf": "1"
                },
                "3": {
                    "class": "Body",
                    "x_1": 28,
                    "y_1": 321,
                    "x_2": 312,
                    "y_2": 932,
                    "isPartOf": "1"
                }
            },
            "00002": {
                "1": {
                    "class": "Horse",
                    "x_1": 1,
                    "y_1": 1,
                    "x_2": 123,
                    "y_2": 124,
                    "isPartOf": ""
                },
                "2": {
                    "class": "Muzzle",
                    "x_1": 23,
                    "y_1": 23,
                    "x_2": 44,
                    "y_2": 44,
                    "isPartOf": "1"
                },
                "3": {
                    "class": "Tail",
                    "x_1": 28,
                    "y_1": 321,
                    "x_2": 312,
                    "y_2": 932,
                    "isPartOf": "1"
                }
            }
        }
        self.annotations = {}
        
    def get_parts_ids(self, filename, obj_id):
        try:
            part_ids = self.annotations[filename]['objects'][obj_id]["hasParts"]
            #pdb.set_trace()
            if part_ids == "":
                #print(f"Object {obj_id} in image {filename} does not have parts.")
                return None
            else:
                return part_ids.split(",")
        except KeyError:
            print(f"Annotation file {filename} or object id {obj_id} do not exist.")
            
    def get_whole_ids(self, filename, obj_id):
        try:
            part_id = self.annotations[filename]['objects'][obj_id]["isPartOf"]
            if part_id == "":
                #print(f"Object {obj_id} in image {filename} is a whole object.")
                return None
            else:
                return part_id
        except KeyError:
            print(f"Annotation file {filename} or object id {obj_id} do not exist.")
            
    def get_objects(self, filename):
        try:
            return self.annotations[filename]['objects']
        except KeyError:
            print(f"Annotation file {filename} does not exist.")
            
    def get_obj_class(self, filename, obj_id):
        try:
            return self.annotations[filename]['objects'][obj_id]["class"]
        except KeyError:
            print(f"Annotation file {filename} or object id {obj_id} do not exist.")

    def get_bounding_box(self, filename, obj_id):
        try:
            x_1 = self.annotations[filename]['objects'][obj_id]["x_1"]
            y_1 = self.annotations[filename]['objects'][obj_id]["y_1"]
            x_2 = self.annotations[filename]['objects'][obj_id]["x_2"]
            y_2 = self.annotations[filename]['objects'][obj_id]["y_2"]
            return [x_1, y_1, x_2, y_2]
        except KeyError:
            print(f"Annotation file {filename} or object id {obj_id} do not exist.")

    def load_data(self, split):
        assert split == "test" or split == "trainval", "split should be 'test' or 'trainval'"
        self.annotations = {}
        self.split = split
        print(f"Parsing {split} set ...")
        path_pascal = 'semanticPascalPart' # here put the path where you have the dataset
        with open(os.path.join(path_pascal, split + ".txt")) as csv_file:
            csv_reader = csv.reader(csv_file)
            for image_name in csv_reader:
                #print(f"Processing annotation file {image_name[0]}")
                tree = ET.parse(os.path.join(path_pascal, f"Annotations_{split}", image_name[0].split(".")[0] + ".xml"))
                xml_data = tree.getroot()
                xmlstr = ET.tostring(xml_data, encoding='utf8', method='xml')
                data_dict = dict(xmltodict.parse(xmlstr))
                filename = data_dict['annotation']['filename'].split(".")[0]
                self.annotations[filename] = {}
                self.annotations[filename]['width'] = int(data_dict['annotation']['imagesize']['ncols'])
                self.annotations[filename]['height'] = int(data_dict['annotation']['imagesize']['nrows'])
                self.annotations[filename]['objects'] = {}
                
                # if image with many objects
                if isinstance(data_dict['annotation']['object'], list):
                    # processing each object
                    for i in range(len(data_dict['annotation']['object'])):
                        self.annotations[filename]['objects'][str(i)] = {"class": data_dict['annotation']['object'][i]["name"].lower().capitalize()}
                        xmin, ymin, xmax, ymax = extract_bb_coordinates(i, data_dict['annotation']['object'])
                        self.annotations[filename]['objects'][str(i)]['x_1'] = int(xmin)
                        self.annotations[filename]['objects'][str(i)]['y_1'] = int(ymin)
                        self.annotations[filename]['objects'][str(i)]['x_2'] = int(xmax)
                        self.annotations[filename]['objects'][str(i)]['y_2'] = int(ymax)
                        whole_obj_id = ""
                        part_obj_ids = ""
                        if "ispartof" in data_dict['annotation']['object'][i]['parts']:
                            if data_dict['annotation']['object'][i]['parts']["ispartof"] is not None:
                                whole_obj_id = data_dict['annotation']['object'][i]['parts']["ispartof"]
                        if "hasparts" in data_dict['annotation']['object'][i]['parts']:
                            if data_dict['annotation']['object'][i]['parts']["hasparts"] is not None:
                                part_obj_ids  = data_dict['annotation']['object'][i]['parts']["hasparts"]
                        self.annotations[filename]['objects'][str(i)]['isPartOf'] = whole_obj_id
                        self.annotations[filename]['objects'][str(i)]['hasParts'] = part_obj_ids
                else:
                    self.annotations[filename]['objects'][str(0)] = {"class": data_dict['annotation']['object']["name"].lower().capitalize()}
                    self.annotations[filename]['objects'][str(0)]['isPartOf'] = ""
                    self.annotations[filename]['objects'][str(0)]['hasParts'] = ""
                    xmin, ymin, xmax, ymax = extract_bb_coordinates(-1, data_dict['annotation']['object'])
                    self.annotations[filename]['objects'][str(0)]['x_1'] = int(xmin)
                    self.annotations[filename]['objects'][str(0)]['y_1'] = int(ymin)
                    self.annotations[filename]['objects'][str(0)]['x_2'] = int(xmax)
                    self.annotations[filename]['objects'][str(0)]['y_2'] = int(ymax)

    def toYOLO(self, name, split, classes, part_hierarchy, class_hierarchy):
        dictionnary_number_to_class, dictionnary_class_to_number = associate_number_to_class_and_class_to_number(classes)
        coded_part_hierarchy = associate_codes_to_hierarchies(dictionnary_class_to_number, part_hierarchy)
        coded_class_hierarchy = associate_codes_to_hierarchies(dictionnary_class_to_number, class_hierarchy)

        # class_codes = get_yolo_class_codes()
        # refined_classes_codes = get_refined_classes()
        for filename in self.annotations.keys():
            yolo_filename = os.path.join(name, f"Annotations_{split}", filename + ".txt")
            km_filename = os.path.join(name, f"Annotations_{split}", filename + ".km")
            width = self.annotations[filename]['width']
            height = self.annotations[filename]['height']
            with open( yolo_filename, 'w', encoding='utf-8') as yolo_file:
                with open( km_filename, 'w', encoding='utf-8') as km_file:
                    for object_id in self.annotations[filename]['objects'].keys():
                        object_class = dictionnary_class_to_number[self.get_obj_class(filename, object_id)]
                        x_lu, y_lu, x_rb, y_rb = self.get_bounding_box( filename, object_id)
                        if ((x_lu == x_rb) or (y_lu == y_rb)):
                            print(f"Object {object_id} in image {filename} is too small (left upper ({x_lu}, {y_lu}) right bottom ({x_rb}, {y_rb})")
                        x_c = float(x_rb + x_lu) / (2.0 * width)
                        y_c = float(y_rb + y_lu) / (2.0 * height)
                        w = float(x_rb - x_lu) / width
                        h = float(y_rb - y_lu) / height
                        yolo_line = str(object_class) + ' ' + str(x_c) + ' ' + str(y_c) + ' ' + str(w) + ' ' + str(h) + '\n'
                        yolo_file.write( yolo_line)
                        km_line = str(object_class)

                        # Handling specialization / generalization relations
                        km_line = km_line + ' ' + add_refined_classes(coded_class_hierarchy, object_class)
                        container = self.get_whole_ids(filename, object_id) 
                        
                        # Handling composition / decomposition relations 
                        if (container != None):
                            while (container != None):
                                container_class = dictionnary_class_to_number[self.get_obj_class(filename, container)]
                                km_line = km_line + ' ' + str(container_class)
                                km_line = km_line + ' ' + add_refined_classes(coded_class_hierarchy, container_class)
                                container = self.get_whole_ids(filename, container)
                        km_line = km_line + '\n'
                        km_file.write(km_line)

    def toRDF(self, name=""):
        print("RDF conversion ...")
        pasPart_namespace = Namespace("http://example.org/pasPart/")
        wordnet_yago_alignment = {}
        with open('ontology/WordNet_Yago_alignment.tsv') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                wordnet_yago_alignment[row['PASCAL-Part_class'].lower().capitalize()] = [row['WDsynset'], row['YagoConcept']]

        g = Graph()
        pas_part_IRI = "https://dkm.fbk.eu/ontologies/semanticPASCALPart/"
        g.bind("https://dkm.fbk.eu/ontologies/semanticPASCALPart", pas_part_IRI)
        pof_uri_ref = URIRef(pasPart_namespace.isPartOf)
        hasParts_uri_ref = URIRef(pasPart_namespace.hasParts)
        g.add((pasPart_namespace.x_1, RDF.type, OWL.DatatypeProperty))
        g.add((pasPart_namespace.y_1, RDF.type, OWL.DatatypeProperty))
        g.add((pasPart_namespace.x_2, RDF.type, OWL.DatatypeProperty))
        g.add((pasPart_namespace.y_2, RDF.type, OWL.DatatypeProperty))
        g.add((pof_uri_ref, RDF.type, OWL.ObjectProperty))
        g.add((hasParts_uri_ref, RDF.type, OWL.ObjectProperty))
        g.add((hasParts_uri_ref, OWL.inverseOf, pof_uri_ref))
        for filename in self.annotations.keys():
            for obj_id in self.annotations[filename].keys():
                obj_class = self.get_obj_class(filename, obj_id)
                bb = self.get_bounding_box(filename, obj_id)
                obj_URI = URIRef(f"{pas_part_IRI}{filename}_{obj_class}_{obj_id}")
                class_URI = URIRef(f"{pas_part_IRI}{obj_class}")
                whole_id = self.get_whole_ids(filename, obj_id)
                if whole_id is not None:
                    part_class = self.get_obj_class(filename, whole_id)
                    whole_URI = URIRef(f"{pas_part_IRI}{filename}_{part_class}_{whole_id}")
                    g.add((obj_URI, pof_uri_ref, whole_URI))
                part_id_list = self.get_parts_ids(filename, obj_id)
                if part_id_list is not None:
                    for part_id in part_id_list:
                        part_class = self.get_obj_class(filename, part_id)
                        part_URI = URIRef(f"{pas_part_IRI}{filename}_{part_class}_{part_id}")
                        g.add((obj_URI, hasParts_uri_ref, part_URI))
                g.add((obj_URI, RDF.type, class_URI))
                g.add((obj_URI, pasPart_namespace.x_1, Literal(bb[0])))
                g.add((obj_URI, pasPart_namespace.y_1, Literal(bb[1])))
                g.add((obj_URI, pasPart_namespace.x_2, Literal(bb[2])))
                g.add((obj_URI, pasPart_namespace.y_2, Literal(bb[3])))
                g.add((obj_URI, pasPart_namespace.hasWordnetId, Literal(wordnet_yago_alignment[obj_class][0])))
                g.add((obj_URI, pasPart_namespace.hasImageName, Literal(filename)))
                g.add((obj_URI, pasPart_namespace.hasYagoConcept, URIRef(wordnet_yago_alignment[obj_class][1])))
        print("Saving RDF file ...")
        g.serialize(destination=f"semantic-PASCAL-Part_{name}.rdf")

if __name__ == '__main__':
    ann = PASCALPart_annotations()
    #ann.load_data(split="test")
    #ann_rdf = ann.toRDF("test")
    ann.load_data(split="trainval")
    ann.toYOLO("semanticPascalPart","trainval")
    
    ann = PASCALPart_annotations()
    ann.load_data(split="test")
    ann.toYOLO("semanticPascalPart","test")
