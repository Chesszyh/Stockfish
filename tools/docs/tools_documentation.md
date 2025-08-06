# Stockfish Tools Documentation

This document provides an overview of the various tools available in the Stockfish tools repository. These tools are primarily used for generating, manipulating, and validating training data for the Stockfish chess engine.

## C++ Tools (in `src/tools`)

These tools are invoked using the `./stockfish` executable followed by a command.

### `convert`

Converts training data between different formats.

*   **`convert_bin_from_pgn_extract`**: Converts data from a PGN extract to binary format.
*   **`convert_bin`**: Converts from text format to binary format.
*   **`convert_plain`**: Converts from binary format to plain text format.

### `generate_training_data`

Generates training data from a given set of opening positions.

### `generate_training_data_nonpv`

Generates training data without using principal variation search.

### `opening_book`

Manages opening books in EPD format.

### `sfen_packer`

Packs and unpacks SFEN (Shogi Fen) positions.

### `stats`

Gathers statistics from training data.

### `transform`

Performs various transformations on training data.

### `validate_training_data`

Validates the integrity of training data.

## Python Scripts (in `script`)

These scripts are executed using the Python interpreter.

### `extract_bin.py`

Extracts a specified number of positions from a binary training data file.

**Usage:**

```bash
python extract_bin.py <filename> <offset> <count> [times]
```

*   `<filename>`: The path to the `.bin` file.
*   `<offset>`: The number of positions to skip from the beginning of the file.
*   `<count>`: The number of positions to extract.
*   `[times]` (optional): The number of times to repeat the extracted positions in the output file.

### `interleave_binpacks.py`

Merges multiple binary training data files into a single file, interleaving the data from the input files.

**Usage:**

```bash
python interleave_binpacks.py <infile1> ... <infileN> <outfile>
```

*   `<infile1>` ... `<infileN>`: The input `.bin` files.
*   `<outfile>`: The name of the output file.

### `pgn_to_plain.py`

Converts PGN files to a plain text format that can be used with the `learn convert_bin` command.

**Usage:**

```bash
python pgn_to_plain.py --pgn "<pgn_files>" --start_ply <ply> --output <output_file>
```

*   `--pgn`: A glob pattern for the input PGN files (e.g., `"data/*.pgn"`).
*   `--start_ply`: The ply number to start extracting positions from.
*   `--output`: The name of the output text file.

### `shuffle_binpack.py`

Shuffles the positions in a binary training data file.

**Usage:**

```bash
python shuffle_binpack.py <infile> <outfile> [split_count]
```

*   `<infile>`: The input `.bin` file.
*   `<outfile>`: The output `.bin` file.
*   `[split_count]` (optional): The number of files to split the shuffled data into.
