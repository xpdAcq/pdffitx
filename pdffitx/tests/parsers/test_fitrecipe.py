import pprint

import mongomock
from mongomock.collection import Collection

from pdffitx.parsers.atoms import dict_to_atoms, dict_to_atoms2
from pdffitx.parsers.fitrecipe import recipe_to_dict, recipe_to_dict2


def test_recipe_to_dict(optimized_recipe):
    client = mongomock.MongoClient()
    collection: Collection = client.db.collection
    dct = recipe_to_dict(optimized_recipe)
    pprint.pprint(dct)
    # test db friendly
    collection.insert_one(dct)
    dict_to_atoms(collection.find_one())


def test_recipe_to_dict2(optimized_recipe):
    client = mongomock.MongoClient()
    collection: Collection = client.db.collection
    dct = recipe_to_dict2(optimized_recipe)
    pprint.pprint(dct)
    collection.insert_one(dct)
    dict_to_atoms2(collection.find_one())
    assert 'eq' in list(dct['conresults'][0].keys())
