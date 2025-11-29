for file in $(find examples -type f -name "*.ir"); do
    python3 filecheck.py "$file"
done
