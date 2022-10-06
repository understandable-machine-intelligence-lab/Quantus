# Clean up.
> make clean

# Build.
> make html

# Convert every .rst file under the docs directory
> rst2myst convert docs/**/*.rst

# Create API docs! Read more here: https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html
# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html

# Solution, generate this in docs
> sphinx-apidoc -o docs_api ../quantus --module-first -f --separate

# Solution, make html here.
> make html

# View edits
http://localhost:63342/Projects/quantus/docs/build/html/index.html#