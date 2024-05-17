set -e # Exit immediately if a command exits with a non-zero status.
set -u # Treat unset variables as an error when substituting.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

python3 $SCRIPT_DIR/build_dictionary.py \
    --dicts \
        $SCRIPT_DIR/download/de-en.txt \
        $SCRIPT_DIR/download/de-fr.txt \
        $SCRIPT_DIR/download/en-de.txt \
        $SCRIPT_DIR/download/en-fr.txt \
        $SCRIPT_DIR/download/en-ja.txt \
        $SCRIPT_DIR/download/en-zh.txt \
        $SCRIPT_DIR/download/fr-de.txt \
        $SCRIPT_DIR/download/fr-en.txt \
        $SCRIPT_DIR/download/ja-en.txt \
        $SCRIPT_DIR/download/zh-en.txt \
    --output_file $SCRIPT_DIR/dictionary.txt \
    --add_meta_symbols

python3 $SCRIPT_DIR/build_dictionary.py \
    --dicts \
        $SCRIPT_DIR/download/de-en.txt \
        $SCRIPT_DIR/download/de-fr.txt \
        $SCRIPT_DIR/download/en-de.txt \
        $SCRIPT_DIR/download/en-fr.txt \
        $SCRIPT_DIR/download/en-ja.txt \
        $SCRIPT_DIR/download/en-zh.txt \
        $SCRIPT_DIR/download/fr-de.txt \
        $SCRIPT_DIR/download/fr-en.txt \
        $SCRIPT_DIR/download/ja-en.txt \
        $SCRIPT_DIR/download/zh-en.txt \
    --output_file $SCRIPT_DIR/dictionary_no_meta.txt