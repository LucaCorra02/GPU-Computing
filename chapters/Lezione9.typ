#import "../template.typ": *

= Lezione 9

I modelli sono molto bravi ad imparare da poco (attraverso esempi). Tuttavia c'è da considerare il problema dell'underfitting.

Il modello impara una dstribuzione di probabil

== Log - Likelihood

//riguardare
== Multiclass

Z è un tensore che produce risultati per 3 classi (batch = 2). Sono dei logit, output del modello, e devono ancora diventare delle probabilità.
I valori di verità $y$ sono i due valori associati al batch.

Vogliamo andare a massimizzare la verosomiglianza (sia penalizzante quando risponde male). Funzione ` crossentropyloss`.

A partire dai logit, applichiamo la cross-entropy applicando $"log-softmax" + "NLL"$.

Ci aspettiamo dei risultati non sorprendenti (alta probabilità se la stima è giusta, basse se è sbagliata).

== Regressione

La funzione ha un output infinito. (da + infinito a meno infinito). Le funzioni di regressione tipiche sono quelle che si usano in ambito euclideo.

Se $x$ appartiene ad un certo spazio vettoriale $R^D$, $b$ è un reale. Vogliamo imparare un iperpiano che ripartisce lo spazio in regioni.

Se supponiamo di avere due dimensioni $x_1$ e $x_2$ (tutti i punti appartengono a $R^2$). Nel caso lineare è una retta. L'obbiettivo è ricavare i valori $W$ e $b$ in modo tale che quando ho dei casi _positivi_ (stanno nella regione $R_1$), ho la classe $C_1$.
I punti che giaciono sulla retta sono i punti che soddisfano l'equazione $y = f(x) = 0$

A seconda del $w$ cambia l'iperpiano, dobbiamo trovare il parametro ideale che separa in modo corretto le due regioni:
- Se $x in R_1$ allora $f(x) > 0$ (funzione positiva)
- Se $x in R_2$ allroa $f(x) > 0$

#nota()[
  Un iperpiano ha sempre $D-1$ dimensioni
]

#attenzione()[
  Il modello funziona solo se i dati sono linearmenti separabili
]

== Modello probabilistico discriminativo

Dato un $x$ vogliamo modellare la probabilità della classe C_k per x (per ogni k, ovvero classe):
Funzione discriminante:
$
  y(x): R^D -> R
$
//riguardare


== SoftMax

Funzione che ci permette di rispondere per casi di due o più classi

Vantaggi di modelli discriminativi:
- Meno parametri (più efficace)
- Migliori prestazioni in generale

== Esempio rigressione non lineare con MLP

Un mlp è caratterizato da:
- Numero di livelli
- Quale funzioni di attivazione non lineari
- Quale funzioni di ottimizzazioni utilizzar

#esempio()[
  Dato un dataset $(x_i,y_i)_(i=1)^N$

  I dati vengono presi nell'intervallo $U[-2,2]$, la funzone di imparare è :
  $
    y = f(x) = x^3-2.5x^2+25 sin(2x) + epsilon
  $
  In genere otteniamo delle coppie di dati osservate $(x,y)$ che hanno del romure. Il rumore lo rappresentiamo in modo artificioso con una variabile aleatorie $epsilon$ che segue una distribuzione uniforme:
  $
    epsilon tilde N(0, sigma^2)
  $
  Il mdoello dato un $x$ mai visto deve rispondere in base a cioò che ha appreso coerentemente
  ```Python
    n = 200 #numero osservazioni
    yNormal = torch.distribuzions.Normal(loc=0.0, scale=10)
    yNoise = yNormal.sample([n])

    #aggiugnere

  ```
  #nota()[
    Il risultato non è mai una funzione precisa continua, ma Solitamente delle ossservazioni
  ]
  == Caso con un neurone in input
  La dimensione del daro è 1 (il batch è 200 dati da 1). il numero di layer è almeno due (si tratta di un bias induttivo).

  Il modello deve definire i parametri theta con una funzione $f(x)$ a partità di ingresso (circa ovviamente).

  Per quanto riguarda la funzione di Loss uso un MSE, minimizzare la discrepanza tra la risposta del modello e quello che ci aspettiamo:
  $
    min_Theta L_theta(x)
  $
  #nota()[
    Non vado a minimizzare $x$, ma vado a minimizzare $Theta$
  ]
  Per quanto riguarda la procedura di training:
  - inizializzare i modelli dei parametri
  - ripeterlo per epoche diverse
  - gradiente discesa

  Il training data sono $x_i,y_i$. è importante fguardare la smoothness dei valori, per vedere se durante il training per vedere se l'architettura è quella corretta.

  Più neuroni aggiungo (dimensione del layer) più ho costo sia di computazione che di parametri che devono ottimizzare. Inoltre potrei avere dei problemi di *overfitting*:
  - Se ho una funzione da imparare (ho pochi dati a disposizione, i puntini)
  - Un mdoello che si comporta bene è un modello che all'incirca segue la funzione (linea gialla), ho un polinomio di grado basso
  - Tuttavia se ho un polinomio di grado alto (più parametri), potrei avere una situazione del modello azzurro. è perfetta sul dataset di training, ma non è in grado di generalizzare

  === Creazione del dataset
  Quando ho un dataset è wrappato attraverso una classe, con dei metodi che forniscono un iteratore sugli elementi del dataset.

  I tentori x e y sono i dati ossservati, lui li ingloba e ci crea un wrapper. In questo modo con $x_i, y_i = d[i]$ posso indicizzare

  `DataLoader(d, batch_size = 25, shuffle=True)`
  Dato il riferimento, diciamo come suddividerlo in gruppi di dati `batch` e se fare o  meno uno shuffleling.

  `shuffle` = se visito il dataset più volte voglio che sia affrontato a batch in ordine diverso. In questo modo il modello visiterà più volte il dataset (epoce) ogni volta in modo diverso. Permette di evitare overfitting.

  Nel nostro caso la $X$ e la $Y$ sono 25 vettori $in R^(25 times 1)$.

  //aggiungere codice

  Risultat:
  - Gli errori tendono a crearsi sugli estremi della funzioen (dati meno concetrati nel dataset sugli estremi)
]

Per quanto riguarda i parametri delle funzioni Linear possiamo passare un oggeto parametro (anichè solo un tensore, facendolo diventare un oggetto pramateo, utile per back propagation). Guardare classe ` nn.Parameters`

`torch.nn.functonal` = per fare operazioni di dettaglio all'interno del forward.

== Rete per la classificazione esempio

Esempi di dati a cerchio divisi in due classi. Vogliamo un classificatore binario (classe 0 e classe 1). (gli algoritmi di clustering in questo caso impazzisono, le classi non sono facilmente isolabili)

=== Dati di input
Abbiamo due dati, $X$ e $Y$, dove la y sara delle coordinate 2D del punto nello spazio e Y sarà 0 o 1 in base alla classe di appartenenza.

I dati vengono splittati in train e test, 80% train e 20% test. Il training set viene estratto a caso dal dataset. A tale scopo utiliziamo la funzione `train_test_split`.

Il modello zeresimo è molto semplice, un unico layer nascosco che prendi in input dati a due dimensioni (coordinate) e li espande a 5 dimensioni. Successivamente l'ouptu dovra essere a una dimensione.

=== Loss
Viene utilizzata una bhi //riguardare

== Training
i logits vengono prodotti, trasformati in probabilità le probabilità vengono arrotondate e infine viene ottenuta l'etichetta.

#nota()[
  Se l'accuratezza è del 50% è molto brutto. Perchè è random guess è come se stessimo lanciando una moneta per decidere la classe
]

Il modello in questo caso creera un iperpiano, tuttavia da dei risultati pessimi. In quanto si tratta semplicemente di una retta
