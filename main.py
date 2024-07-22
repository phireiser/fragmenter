import requests, zipfile, io
from fragmentation import Fragmentation

import json


def loadMsLibraryMonaJson2spectrum4Smiles(file) -> list:
    """
    Loads a MoNA JSON File and builds a computable data reduced datastructure from the file passed

    Args:
        file: MoNA file; a .read()-supporting file-like object
    
    Returns:
        list: data reduced datastructure
    """
    mona_data = json.load(file)
    
    mona = []

    for data in mona_data:
        names = data['compound'][0]['metaData']
        smiles = ''
        for name in names:
            if name['name'] == 'SMILES':
                smiles = name['value']
        
        spectrum = []
        # data['spectrum'] = "61.98574522280524:0.12206526 68.05251722280524:0.05164368 70.06191722280523:0.05164368 77.96997522280525:0.05164368 97.08282922280524:0.18309789 114.05580222280524:0.48826213 127.09312222280523:0.05164368"
        pairs = data['spectrum'].split(' ')
        for pair in pairs:
            tmp_lst = pair.split(':')
            mass_intensity = [float(i) for i in tmp_lst]
            spectrum.append(tuple(mass_intensity))

        molecule = tuple([smiles, spectrum])
        mona.append(molecule)

    return mona






zip_url_MoNA_export_Dipeptide_neg = 'https://mona.fiehnlab.ucdavis.edu/rest/downloads/retrieve/1fdac070-821c-418a-80e5-c8a0b2babc50'
zip_url_MoNA_export_LC_MS_MS_Positive_Mode = 'https://mona.fiehnlab.ucdavis.edu/rest/downloads/retrieve/7609a87b-5df1-4343-afe9-2016a3e79516'

response = requests.get(zip_url_MoNA_export_Dipeptide_neg)
response.raise_for_status()

zip_file = zipfile.ZipFile(io.BytesIO(response.content))

file_name = zip_file.namelist()[0]
with zip_file.open(file_name) as extracted_file:
    # Read the file content into memory
    file_content = extracted_file.read()

mona_data = loadMsLibraryMonaJson2spectrum4Smiles(io.BytesIO(file_content))

for mona_molecule in mona_data[:14]: # slice of 14 samples
    mona_smiles = mona_molecule[0]
    mona_spectrum = mona_molecule[1]

    x = Fragmentation(mona_smiles, mona_spectrum, 20)
    x.plotNormalizedSpectrum()