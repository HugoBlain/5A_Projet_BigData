import csv

tsv_file = open("wikirank-fr.tsv")
read_tsv = csv.reader(tsv_file, delimiter="\t")
articles = dict()
for row in read_tsv:
  # We skip strange lines (headers) that do not correspond to articles
  if row[0] == 'page_id' : continue

  id = int(row[0])
  name = row[1]
  quality = float(row[2])
  pop = float(row[3])
  authors = float(row[4])
  if id in articles:
    dejavu = articles[id]
    if dejavu == (id, name, quality, pop, authors):
#       print ('Doublon normal pour id = ', id)
       pass
    else:
      print ('Doublon etrange !!! pour id = ', id)
  else:
    articles[id] = (id, name, quality, pop, authors)

print ('Ex 6. Nombre d\'articles uniques ', len (articles))

print ()
print ('Ex 7. Trier par popularité et afficher le TOP20')
# Unix
# cat wikirank-fr.tsv | sort | uniq | sort -t$'\t' -k4 -r -n | head -n20
srtd = sorted (articles.values(), key = lambda x : x[3], reverse = True)
for i in range (20):
  print (srtd[i])

print ()  
print ('Ex 8. TOP 20 des articles les moins populaires')
# Unix
# cat wikirank-fr.tsv | sort | uniq | sort -t$'\t' -k4 -n | head -n20
for i in range (len(srtd) - 20, len(srtd)):
  print (srtd[i])
  


print ()
print ('''Ex 9. Visualisation. Il y a trop de points ~= 2m.
Il sera difficile de tous les dessiner, mais on peut essayer.
''')
import matplotlib as mpl
import matplotlib.pyplot as plt

popularity = [ line[3] for line in srtd]
authors_number = [ line[4] for line in srtd]

from numpy import corrcoef
c = corrcoef(popularity, authors_number)[0][1]

plt.scatter(popularity, authors_number )
plt.xlabel("Popularity")
plt.ylabel("# authors")
plt.title("Scatter plot. Pearson correlation = {}".format(c))
plt.show()

print('''ou bien 2d histogramme''')
plt.hist2d (popularity, authors_number, bins = [30, 30], norm = mpl.colors.LogNorm())
plt.colorbar()
plt.xlabel("Popularity")
plt.ylabel("# authors")
plt.title("Density plot with logarithmic colors. Pearson correlation = {}".format(c))
plt.show()


print('''ou en échelle logarithmique''')
plt.scatter(popularity, authors_number)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Popularity")
plt.ylabel("# authors")
plt.title("Log-log plot. Pearson correlation = {}".format(c))
plt.show()


print("""Ex. 10. Exemple de caractéristiques aberrantes. Articles très populaire mais peu d'auteurs""")
for a in srtd:
  if a[3] > 5000 and a[4] < 100:
    print (a)

print("""Articles peu populaire mais trop d'auteurs""")
for a in srtd:
  if a[3] < 2000 and a[4] > 70:
    print (a)
