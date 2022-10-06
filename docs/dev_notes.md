
# Create API docs! Read more here: https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html
# Or here. https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
> cd docs
> sphinx-apidoc -o source/docs_api ../quantus --module-first -f --separate
> make clean
> make html

# View edits
http://localhost:63342/Projects/quantus/docs/build/html/index.html#

# Convert every .rst file under the docs directory
> rst2myst convert docs/**/*.rst
