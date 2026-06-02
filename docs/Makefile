PYTHON = python3
ROOT   = ..
GALLERY_STAMP = examples/_generated_galleries/.stamp

.PHONY: help install gallery gallery-force serve build clean clean-gallery clean-site

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "  install       Install documentation dependencies"
	@echo "  gallery       Generate galleries only if not already present"
	@echo "  gallery-force Regenerate galleries unconditionally"
	@echo "  serve         Build gallery if needed, then serve docs with live reload"
	@echo "  build         Build gallery if needed, then build static site"
	@echo "  clean         Remove all generated files (gallery + site)"
	@echo "  clean-gallery Remove only generated gallery pages"
	@echo "  clean-site    Remove only the built site directory"

install:
	pip install -r $(ROOT)/requirements.txt

$(GALLERY_STAMP):
	cd $(ROOT) && $(PYTHON) scripts/create_gallery.py
	touch $(GALLERY_STAMP)

gallery: $(GALLERY_STAMP)

gallery-force:
	cd $(ROOT) && $(PYTHON) scripts/create_gallery.py
	touch $(GALLERY_STAMP)

serve: $(GALLERY_STAMP)
	cd $(ROOT) && mkdocs serve

build: $(GALLERY_STAMP)
	cd $(ROOT) && mkdocs build

clean: clean-gallery clean-site

clean-gallery:
	rm -rf examples/_generated_galleries
	rm -rf coding/_generated_galleries
	rm -rf catalog/_generated_galleries
	rm -f $(GALLERY_STAMP)

clean-site:
	rm -rf $(ROOT)/site
