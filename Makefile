SAMPLES=10000
YEARS=2014 2015 2016 2017 2018 2019 2020

setup:
	mkdir -p data plots
	$(foreach val, $(YEARS)$, mkdir plots/$(val);)

run_base_analysis: python/load_data.py python/standard_model.py python/multiyear_analysis.py
	# Load data
	python python/load_data.py -y 2000 --savedir data/
	# Evaluate model for each year
	$(foreach val, $(YEARS)$, python python/standard_model.py -y $(val) -s $(SAMPLES);)
	# Run year comparison
	python python/multiyear_analysis.py data/ $(SAMPLES)
