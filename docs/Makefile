                                                                                                                                                               
all: paper

clean:
	-rm -f *.aux *.log *.dvi *.blg *.out *.bbl ms.pdf

# ----------------------------------------------------------------------------

pdf: ms.pdf 

paper: ms.pdf

ms.pdf: ms.tex macros.tex
	pdflatex ms.tex
	pdflatex ms.tex

# ----------------------------------------------------------------------------

