def mat2pdb(input_data):
    """
    This function creates a PDB from coordinate data. The output format follows
    the official PDB format documentation.
    
    Required Inputs (as part of input_data dict):
    - X: orthogonal X coordinate data (angstroms)
    - Y: orthogonal Y coordinate data (angstroms)
    - Z: orthogonal Z coordinate data (angstroms)
    
    Optional Inputs:
    - outfile: output file name (default: 'mat2PDB.pdb')
    - recordName: record name of atoms (default: 'ATOM')
    - atomNum: atom serial number (default: sequential)
    - atomName: name of atoms (default: 'CA')
    - altLoc: alternate location indicator (default: ' ')
    - resName: name of residue (default: 'MET')
    - chainID: protein chain identifier (default: 'B')
    - resNum: residue sequence number (default: sequential)
    - occupancy: occupancy factor (default: 1.00)
    - betaFactor: beta factor, temperature (default: 0.00)
    - element: element symbol (default: 'O')
    - charge: atomic charge (default: ' ')
    """

    # Required: X, Y, Z coordinates
    if 'X' not in input_data or 'Y' not in input_data or 'Z' not in input_data:
        print("XYZ coordinate data is required to make a PDB. Exiting...")
        return
    
    X = input_data['X']
    Y = input_data['Y']
    Z = input_data['Z']
    
    if len(X) != len(Y) or len(X) != len(Z):
        print("XYZ coordinate data is not of equal lengths. Exiting...")
        return

    # Optional inputs with default values
    input_data.setdefault('outfile', 'mat2PDB.pdb')
    input_data.setdefault('recordName', ['ATOM'] * len(X))
    input_data.setdefault('atomNum', list(range(1, len(X) + 1)))
    input_data.setdefault('atomName', ['CA'] * len(X))  # Default is CA
    input_data.setdefault('altLoc', [' '] * len(X))
    input_data.setdefault('resName', ['MET'] * len(X))  # Default is MET
    input_data.setdefault('chainID', ['B'] * len(X))  # Default chain ID B
    input_data.setdefault('resNum', list(range(1, len(X) + 1)))
    input_data.setdefault('occupancy', [1.0] * len(X))
    input_data.setdefault('betaFactor', [0.0] * len(X))
    input_data.setdefault('element', ['O'] * len(X))  # Default element is O (oxygen)
    input_data.setdefault('charge', [' '] * len(X))

    # Get all variables
    outfile = input_data['outfile']
    recordName = input_data['recordName']
    atomNum = input_data['atomNum']
    atomName = input_data['atomName']
    altLoc = input_data['altLoc']
    resName = input_data['resName']
    chainID = input_data['chainID']
    resNum = input_data['resNum']
    occupancy = input_data['occupancy']
    betaFactor = input_data['betaFactor']
    element = input_data['element']
    charge = input_data['charge']

    # Fix atomName spacing
    atomName = [f'{name:<3}' for name in atomName]

    # Open file for writing PDB
    #print(f'Outputting PDB in file {outfile}')
    with open(outfile, 'w') as file:
        # Write each atom's data into the PDB file
        for n in range(len(atomNum)):
            file.write(f"{recordName[n]:<6}{atomNum[n]:>5} {atomName[n]:<4}{altLoc[n]:<1}"
                       f"{resName[n]:<3} {chainID[n]:<1}{resNum[n]:>4}    {X[n]:>8.3f}{Y[n]:>8.3f}{Z[n]:>8.3f}"
                       f"{occupancy[n]:>6.2f}{betaFactor[n]:>6.2f}          {element[n]:>2}{charge[n]:>2}\n")
            
            # Display progress for large datasets
            if n % 400 == 0 and n != 0:
                print(f"   {100 * n / len(atomNum):.2f}%")

        # Write CONECT records
        for n in range(len(atomNum) - 1):
            file.write(f"CONECT{atomNum[n]:>5}{atomNum[n + 1]:>5}\n")

        # End the PDB file
        file.write("END\n")

    print("100.00% done! Closing file...")