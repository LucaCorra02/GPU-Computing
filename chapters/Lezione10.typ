#import "../template.typ": *

= PyTorch autograd

== Fondamenti Matematici del Gradiente
Il *gradiente* è un operatore differenziale che si applica a funzioni scalari definite su *spazi vettoriali*. Data una funzione:
$
  f: R^D -> R
$
che mappa un vettore di $D$ dimensioni in uno scalare, il gradiente di $f$ è definito come:
$
  nabla f: R^D -> R^D
$

Il gradiente produce un *vettore* che contiene tutte le *derivate parziali* della funzione rispetto a ciascuna variabile (si deriva rispetto a una varianbile alla volta, tenendo le altre costanti):
$
  nabla f(x) = vec((partial f)/(partial x_1), (partial f)/(partial x_2), dots.v, (partial f)/(partial x_D))
$

#nota()[
  Il gradiente trasforma una funzione scalare in un *campo vettoriale*: *ad ogni punto $x$* dello spazio viene associato un vettore *$nabla f(x)$*.

  Nel contesto del deep learning:
  - $f$ rappresenta la *funzione di loss*
  - $x$ rappresenta i *parametri* del modello (pesi e bias)
  - $nabla f(x)$ indica come modificare i parametri per ridurre la loss
]

=== Derivate Parziali

Una *derivata parziale* misura il tasso di variazione di una funzione rispetto a *una singola variabile*, mantenendo costanti tutte le altre.

Data $f(x_1, x_2, dots, x_D)$, la derivata parziale rispetto a $x_i$ è:
$
  (partial f)/(partial x_i) = lim_(h -> 0) (f(x_1, dots, x_i + h, dots, x_D) - f(x_1, dots, x_i, dots, x_D))/h
$

Supponendo di lavorare in uno spazio vettoriale a *due dimensioni* ($R^2$, con $D = 2$), possiamo dare un'interpretazione geometrica alle derivate parziali:

#figure[
  #image("../assets/DerivateParziali2d.png", width: 80%)

  Visualizzazione 3D di una funzione $f(x,y)$ con le derivate parziali nel punto rosso $(1,1)$.
]

In particolare, nell'immagine possiamo oservare che:
- *spazio vettoriale*: rappresentato dagli assi $(x,y)$. Ogni punto $(x,y)$ rappresenta una coppia di valori per le variabili $x$ e $y$.

- funzione *$f(x,y)$*: per ogni punto $(x,y)$, la funzione associa un valore scalare $f(x,y)$. Questo valore rappresenta l'*altitudine* ($z$) della superficie sopra quel punto.

Consideriamo ora il punto $mr("rosso")$ di coordinate $(1,1)$. Se volessimo _quanto è ripida la salita_, potremo utilizzare la *derivata parziale*. Essa indica lo spostamento rispette alle due direzioni principali $(x,y)$. Nell'immagine:


- $mb("Linea Blu")$ (Derivata rispetto a $x$): La curva viene tagliata lungo l'asse $x$ (linea blu tratteggiata). La linea blu spessa è la pendenza esatta in quel punto muovendosi solo lungo $x$, indica il *tasso di variazione di $f$* se facciamo un piccolo passo verso Est (aumentando $x$). Se la linea blu è inclinata verso l'alto, significa che aumentando $x$ aumenta anche $f$ (salita).

- $mg("Linea Verde")$ (Derivata rispetto a $y$): È la stessa cosa, ma nell'altra direzione (asse $Y$). La linea verde spessa indica il tasso di variazione di $f$ se facciamo un piccolo passo verso Nord (aumentando $y$). Se la linea verde è inclinata verso l'alto, significa che aumentando $y$ aumenta anche $f$ (salita).

==== Il Gradiente: Combinazione delle Derivate Parziali

Il *gradiente* $nabla f(x, y)$ è un *vettore* che combina entrambe le derivate parziali:
$
  nabla f(x, y) = vec((partial f)/(partial x), (partial f)/(partial y))
$

Nel contesto della visualizzazione 3D:
- Le derivate parziali (linee blu e verde) misurano la pendenza lungo le *direzioni degli assi*
- Il gradiente combina queste informazioni in un *unico vettore 2D* nel piano $(x, y)$
- Questo vettore indica la *direzione di massima salita* sulla superficie

#figure(
  cetz.canvas({
    import cetz.draw: *

    // Assi cartesiani
    line((-0.5, 0), (4.5, 0), mark: (end: ">"), stroke: 1pt)
    content((4.7, 0), $x$, anchor: "west")
    line((0, -0.5), (0, 3.5), mark: (end: ">"), stroke: 1pt)
    content((0, 3.7), $y$, anchor: "south")

    // Punto di interesse
    let px = 1.8
    let py = 1.5
    circle((px, py), radius: 0.1, fill: red, stroke: none)
    content((px - 0.4, py - .4), text(size: 8pt, $(x_0, y_0)$), fill: red)

    // Derivata parziale rispetto a x (componente orizzontale)
    let dx = 1.0 // lunghezza proporzionale a ∂f/∂x
    line((px, py), (px + dx, py), mark: (end: ">"), stroke: (paint: blue, thickness: 2pt, dash: "dashed"))
    content((px + dx / 2, py - 0.3), text(size: 8pt, $(partial f)/(partial x)$), fill: blue)

    // Derivata parziale rispetto a y (componente verticale)
    let dy = 0.75 // lunghezza proporzionale a ∂f/∂y
    line((px, py), (px, py + dy), mark: (end: ">"), stroke: (paint: green.darken(20%), thickness: 2pt, dash: "dashed"))
    content((px - 0.5, py + dy / 2), text(size: 8pt, $(partial f)/(partial y)$), fill: green.darken(20%))

    // Gradiente (vettore risultante)
    line((px, py), (px + dx, py + dy), mark: (end: ">"), stroke: (paint: red.darken(20%), thickness: 2.5pt))
    content((px + dx / 2 + 0.3, py + dy / 2 + 0.6), text(size: 9pt, weight: "bold", $nabla f$), fill: red.darken(20%))

    // Rettangolo tratteggiato per visualizzare la composizione
    line((px + dx, py), (px + dx, py + dy), stroke: (dash: "dotted", paint: gray))
    line((px, py + dy), (px + dx, py + dy), stroke: (dash: "dotted", paint: gray))

    // Origine
    content((-0.3, -0.3), text(size: 8pt, $O$))

    // Legenda (spostata in basso a destra)
    rect((3.2, 0.2), (rel: (1.4, 0.9)), fill: white.transparentize(10%), stroke: 0.8pt)
    line((3.3, 0.9), (3.6, 0.9), mark: (end: ">"), stroke: (paint: blue, thickness: 1.5pt, dash: "dashed"))
    content((3.65, 0.9), text(size: 7pt, [Comp. $x$]), anchor: "west")
    line(
      (3.3, 0.65),
      (3.6, 0.65),
      mark: (end: ">"),
      stroke: (paint: green.darken(20%), thickness: 1.5pt, dash: "dashed"),
    )
    content((3.65, 0.65), text(size: 7pt, [Comp. $y$]), anchor: "west")
    line((3.3, 0.4), (3.6, 0.4), mark: (end: ">"), stroke: (paint: red.darken(20%), thickness: 2pt))
    content((3.65, 0.4), text(size: 7pt, weight: "bold", [Gradiente]), anchor: "west")
  }),
  caption: [Il gradiente $nabla f$ ($mr("rosso")$) è il vettore che si ottiene combinando le due derivate parziali: la componente $mb("blu")$ $(partial f)/(partial x)$ (orizzontale) e la componente $mg("verde")$ $(partial f)/(partial y)$ (verticale). Nel piano 2D $(x,y)$, il gradiente indica la direzione di massima crescita di $f$.],
)

#nota()[
  *Interpretazione geometrica del gradiente*:

  - *Direzione*: Il gradiente punta nella direzione in cui la funzione cresce più rapidamente
  - *Magnitudine*: La lunghezza del vettore gradiente indica quanto ripida è la salita
  - *Posizione nel piano*: Il gradiente "vive" nel piano $(x, y)$ (piano di base), non nello spazio 3D della superficie
]

==== Perpendicolarità alle Curve di Livello

Un concetto fondamentale è che il gradiente è *perpendicolare* alle curve di livello:

*Curve di livello*: Se proiettiamo la superficie 3D sul piano $(x, y)$ e colleghiamo tutti i punti che hanno la *stessa altitudine* $f(x, y) = c$, otteniamo delle curve chiamate _curve di livello_:
- Se le curve di livello sono vicine tra loro, la salita è ripida (il gradiente sarebbe una freccia lunga).
- Se le curve sono lontane, il terreno è dolce/quasi piatto.

*Perpendicolarità*: In ogni punto, il vettore gradiente $nabla f$ è *perpendicolare* (ortogonale, a 90°) alla curva di livello che passa per quel punto.

#attenzione()[
  *Importante*: La perpendicolarità è nel *piano $(x, y)$*, non nello spazio 3D!

  - Il gradiente è un vettore 2D: $nabla f(x, y) = vec((partial f)/(partial x), (partial f)/(partial y))$
  - Le curve di livello sono curve 2D nel piano $(x, y)$
  - La perpendicolarità si riferisce a queste entità 2D

  Quando visualizziamo la superficie 3D, "proiettiamo" mentalmente il gradiente sul piano di base per vedere la sua perpendicolarità alle curve di livello.
]

#esempio()[
  Consideriamo la funzione quadratica in due variabili:
  $
    f(x_1, x_2) = x_1^2 + 3x_1 x_2 + 2x_2^2
  $

  *Derivata parziale rispetto a $x_1$*:

  Trattiamo $x_2$ come una costante:
  $
    (partial f)/(partial x_1) = 2x_1 + 3x_2
  $

  *Derivata parziale rispetto a $x_2$*:

  Trattiamo $x_1$ come una costante:
  $
    (partial f)/(partial x_2) = 3x_1 + 4x_2
  $

  *Gradiente completo*: Crea uno spazio vettoriale con entrambe le derivate:
  $
    nabla f(x_1, x_2) = vec(2x_1 + 3x_2, 3x_1 + 4x_2)
  $

  #figure(
    cetz.canvas({
      import cetz.draw: *

      // Assi
      line((-3, 0), (3, 0), mark: (end: ">"))
      content((3.3, 0), $x_1$)
      line((0, -2.5), (0, 2.5), mark: (end: ">"))
      content((0, 2.8), $x_2$)

      // Curve di livello per la funzione f(x1,x2) = x1² + 3x1*x2 + 2x2²
      // Questa è una forma quadratica, le curve di livello sono ellissi ruotate
      for level in (1, 3, 6, 10) {
        let points = ()
        for i in range(0, 101) {
          let angle = i * 2 * calc.pi / 100
          // Uso parametrizzazione approssimata per le curve di livello
          let r = calc.sqrt(level) / calc.sqrt(1 + 0.5 * calc.sin(2 * angle))
          let x1 = r * calc.cos(angle)
          let x2 = r * calc.sin(angle)
          // Verifica approssimata che soddisfi l'equazione
          points.push((x1 * 0.8, x2 * 0.7))
        }
        line(..points, stroke: (paint: blue.lighten(40%), thickness: 1pt), close: true)
      }

      // Etichette per le curve di livello
      content((0.8, 0.1), text(size: 7pt, $c_1$), anchor: "south", fill: blue)
      content((1.5, 0.1), text(size: 7pt, $c_2$), anchor: "south", fill: blue)

      // Punti campione e vettori gradiente
      let sample_points = (
        (1, 0.5),
        (0.5, 1),
        (-0.5, 0.8),
        (-1, -0.5),
        (0.8, -0.6),
        (1.5, 0),
        (0, 1.2),
        (-1.2, 0.3),
      )

      for pt in sample_points {
        let (x1, x2) = pt
        // Punto
        circle((x1, x2), radius: 0.06, fill: red, stroke: none)

        // Calcola gradiente: ∇f = (2x₁ + 3x₂, 3x₁ + 4x₂)
        let grad_x1 = 2 * x1 + 3 * x2
        let grad_x2 = 3 * x1 + 4 * x2

        // Scala per visualizzazione
        let scale = 0.15
        let gx = grad_x1 * scale
        let gy = grad_x2 * scale

        // Disegna vettore gradiente
        line((x1, x2), (x1 + gx, x2 + gy), mark: (end: ">"), stroke: (paint: red, thickness: 1.3pt))
      }

      // Origine
      circle((0, 0), radius: 0.08, fill: green, stroke: none)
      content((0.3, -0.2), text(size: 8pt, $(0,0)$), fill: green)

      // Legenda
      rect((-3.8, 1.8), (-1, 2.6), fill: white.transparentize(20%), stroke: 0.5pt)
      line((-3.6, 2.4), (-3.2, 2.4), stroke: (paint: blue.lighten(40%), thickness: 1pt))
      content((-3.0, 2.4), text(size: 7pt, [Curve di livello]), anchor: "west")
      line((-3.6, 2.1), (-3.2, 2.1), mark: (end: ">"), stroke: (paint: red, thickness: 1.3pt))
      content((-3, 2.1), text(size: 7pt, [$nabla f$]), anchor: "west")
    }),
    caption: [Campo vettoriale del gradiente per $f(x_1, x_2) = x_1^2 + 3x_1 x_2 + 2x_2^2$. I vettori rossi rappresentano $nabla f(x_1, x_2) = vec(2x_1 + 3x_2, 3x_1 + 4x_2)$ in vari punti, e sono sempre perpendicolari alle curve di livello blu.],
  )

  In un punto specifico, ad esempio $(x_1, x_2) = (1, 2)$:
  $
    nabla f(1, 2) = vec(2(1) + 3(2), 3(1) + 4(2)) = vec(8, 11)
  $
]

=== Formalizzazione geometrica del gradiente

Il gradiente ha diverse interpretazioni geometriche fondamentali:

1. * Direzione di massima crescita*: Il vettore $nabla f(x_0)$ punta nella direzione in cui la funzione $f$ *cresce più rapidamente* a partire dal punto $x_0$.

2. *Vettore normale alle curve di livello*: Il gradiente in un punto è *perpendicolare* alla curva (o superficie) di livello passante per quel punto.

3. * Magnitudine come tasso di crescita*: Il modulo $||nabla f(x_0)||$ rappresenta il *tasso di crescita massimo* di $f$ in $x_0$.

#esempio()[
  Consideriamo il paraboloide:
  $
    f(x, y) = x^2 + y^2
  $

  Il gradiente è:
  $
    nabla f(x, y) = vec(2x, 2y)
  $

  #figure(
    cetz.canvas({
      import cetz.draw: *

      // Configurazione
      let scale = 1.5

      // Assi
      line((-3, 0), (3, 0), mark: (end: ">"))
      content((3.3, 0), $x$)
      line((0, -3), (0, 3), mark: (end: ">"))
      content((0, 3.3), $y$)

      // Curve di livello (cerchi concentrici)
      for r in (0.5, 1.0, 1.5, 2.0, 2.5) {
        circle((0, 0), radius: r, stroke: (paint: blue.lighten(30%), thickness: 1pt))
      }

      // Etichette curve di livello
      content((0.5, 0), text(size: 7pt, $c_1$), anchor: "south", fill: blue)
      content((1.5, 0), text(size: 7pt, $c_2$), anchor: "south", fill: blue)
      content((2.5, 0), text(size: 7pt, $c_3$), anchor: "south", fill: blue)

      // Punti campione
      let points = ((1, 0.5), (0.7, 1.5), (-1, 1), (-1.5, -0.5), (0.5, -1.5))

      for pt in points {
        let (x, y) = pt
        // Punto
        circle((x, y), radius: 0.08, fill: red, stroke: none)

        // Calcola gradiente: ∇f = (2x, 2y)
        let grad_x = 2 * x * 0.3 // Scala per visualizzazione
        let grad_y = 2 * y * 0.3

        // Disegna vettore gradiente
        line((x, y), (x + grad_x, y + grad_y), mark: (end: ">"), stroke: (paint: red, thickness: 1.5pt))
      }

      // Origine
      circle((0, 0), radius: 0.1, fill: green, stroke: none)
      content((0, -0.4), text(size: 8pt, $(0,0)$), fill: green)

      // Legenda
      rect((-2.8, 2), (-1.2, 2.8), fill: white.transparentize(20%), stroke: 0.5pt)
      line((-2.6, 2.6), (-2.2, 2.6), stroke: (paint: blue.lighten(30%), thickness: 1pt))
      content((-1.9, 2.6), text(size: 7pt, [Curve di livello]), anchor: "west")
      line((-2.6, 2.3), (-2.2, 2.3), mark: (end: ">"), stroke: (paint: red, thickness: 1.5pt))
      content((-1.9, 2.3), text(size: 7pt, [$nabla f$]), anchor: "west")
    }),
    caption: [Campo vettoriale del gradiente $nabla f(x,y) = vec(2x, 2y)$ per il paraboloide $f(x,y) = x^2 + y^2$. I vettori rossi (gradienti) sono perpendicolari alle curve di livello blu e puntano verso l'esterno (crescita).],
  )

  *Osservazioni*:
  - All'origine $(0, 0)$, il gradiente è nullo: $nabla f(0,0) = vec(0, 0)$ (punto di minimo)
  - Nei punti lontani dall'origine, i vettori gradiente sono più lunghi (crescita più rapida)
  - I gradienti puntano *radialmente verso l'esterno* (direzione di massima crescita)
  - Sono tutti *perpendicolari* ai cerchi (curve di livello)
]

=== Curve di Livello (Level Sets)

Le *curve di livello* (o superfici di livello in dimensioni superiori) sono l'insieme dei punti dove la funzione assume un *valore costante*:
$
  cal(L)_c = {x in R^D : f(x) = c}
$

#nota()[
  *Proprietà fondamentale*: Il gradiente $nabla f(x_0)$ in un punto $x_0$ è *ortogonale* alla curva di livello passante per $x_0$.

  Questo significa che il gradiente punta nella direzione che *esce* dalla curva di livello, verso valori crescenti di $f$.
]

#attenzione()[
  Le curve di livello sono fondamentali nell'ottimizzazione:

  - Un *passo di gradient descent* ci sposta da una curva di livello a un'altra con valore inferiore
  - La direzione *perpendicolare* alla curva (il gradiente) è la direzione di *massimo cambiamento*
  - Seguire il gradiente negativo ci porta verso valori decrescenti di $f$ (verso il minimo)
]

=== Teorema della Direzione di Massima Discesa

Si tratta del teorema fondamentale che giustifica l'uso del gradiente nell'ottimizzazione.

#teorema("Massima Discesa")[
  Dato un punto $x_0 in R^D$ e una funzione differenziabile $f: R^D -> R$, tra tutte le direzioni unitarie $d$ (con $||d|| = 1$), la direzione di *massima discesa* (che minimizza $f$ localmente) è data da:
  $
    d^* = -(nabla f(x_0)) / (||nabla f(x_0)||)
  $
  Ovvero, il *negativo del gradiente* indica la direzione in cui la funzione *decresce più rapidamente*.
]

#dimostrazione()[
  L'obbiettivo della dimostrazione è trovare una direzione unitaria $d$ che minimizza la funzione $f$ localmente intorno a $x_0$.

  Per la dimostrazione è necessario considerare lo *sviluppo di Taylor* al primo ordine di $f$ intorno a $x_0$:
  $
    f(x_0 + alpha d) approx f(x_0) + alpha nabla f(x_0)^T d
  $
  dove:
  - $d$ è una direzione unitaria ($||d|| = 1$)
  - $alpha > 0$ è uno step size piccolo


  La nuova _altezza_  della funzione dopo il passo ($f(x_0 + alpha d)$) è data dalla vecchia altezza ($f(x_0)$) più un termine di variazione ($alpha nabla f(x_0)^T d$).

  #nota()[
    Lo sviluppo di Taylor ci dice che per piccoli spostamenti $alpha d$ da $x_0$, la funzione cambia approssimativamente di:
    $
      Delta f approx alpha nabla f(x_0)^T d
    $
  ]
  Per scendere al massimo nella funzione, vogliamo che questo _cambiamento_ sia il numero più negativo possibile. Siccome $alpha$ è positivo e fisso, tutto dipende dal Prodotto Scalare tra il gradiente e la tua direzione:
  $
    nabla f(x_0)^T d
  $
  Il prodotto scalare tra due vettori geometricamente può essere calcolato come.
  $
    u dot v = ||u|| dot ||v|| dot cos(theta)
  $
  Applicando la proprietà del prodotto scalare al nostro caso:
  $
    nabla f(x_0)^T d = ||nabla f(x_0)|| dot ||d|| dot cos(theta)
  $
  Dove:
  - $||d|| = 1$ (direzione unitaria)
  - $||nabla f(x_0)||$ è un numero fisso.

  L'unica variabile è $cos(theta)$, che dipende dall'angolo tra il gradiente e la direzione $d$. Siccome il valore minimo del coseno vale $-1$ quando l'angolo $theta$ vale $180$ gradi, significa che i due vettori sono opposti (puntano in *direzioni esattamente opposte*):

  $
    d^* = -(nabla f(x_0))/(||nabla f(x_0)||)
  $

]

#nota()[
  *Conclusione*: Muoversi nella direzione $-nabla f(x_0)$ garantisce la *massima riduzione* della funzione $f$ in un intorno di $x_0$.

  Questo è il principio alla base del *Gradient Descent*!
]

=== Gradient Descent: Algoritmo di Base

Dato un punto iniziale $x_0 in R^n$ e una funzione obbiettivo $f: R^n -> R$ l'algoritmo di *discesa del gradiente* procede iterativamente. In particolare ad ogni passo:
$
  x_(k+1) = x_k - alpha_k nabla f(x_k)
$
dove:
- *$x_k$*: valore dei parametri all'iterazione $k$
- *$alpha_k > 0$*: *learning rate* (passo di discesa)
- $nabla f(x_k)$: gradiente calcolato in $x_k$

*Algoritmo iterativo*:
1. Inizializza $x_0$ (casualmente o con euristica)
2. Per $k = 0, 1, 2, dots$ fino a *convergenza*:
  - Calcola il gradiente: $g_k = nabla f(x_k)$
  - Aggiorna i parametri: $x_(k+1) = x_k - alpha_k g_k$
  - Controlla convergenza: se $||g_k|| < epsilon$, termina

#nota()[
  *Criteri di convergenza*:
  - *Condizione ideale*: $nabla f(x_k) = bold(0)$ (punto stazionario)
  - *Condizione pratica*: $||nabla f(x_k)|| < epsilon$ per una soglia piccola $epsilon > 0$

  #attenzione()[
    Un gradiente nullo può indicare:
    - Un *minimo locale* (desiderato)
    - Un *massimo locale* (indesiderato)
    - Un *punto di sella* (né minimo né massimo)

    In deep learning, i *punti di sella* sono molto più comuni dei minimi locali in alta dimensione.
  ]
]

#esempio()[
  Minimizziamo $f(x, y) = x^2 + 4y^2$ partendo da $(x_0, y_0) = (3, 2)$.

  *Gradiente*:
  $
    nabla f(x, y) = vec(2x, 8y)
  $

  *Iterazioni con $alpha = 0.1$*:

  - $k=0$:
    - $x_0 = vec(3, 2)$
    - $nabla f(x_0) = vec(6, 16)$
    - $x_1 = vec(3, 2) - 0.1 vec(6, 16) = vec(2.4, 0.4)$

  - $k=1$:
    - $nabla f(x_1) = vec(4.8, 3.2)$
    - $x_2 = vec(2.4, 0.4) - 0.1 vec(4.8, 3.2) = vec(1.92, 0.08)$

  - $k=2$:
    - $nabla f(x_2) = vec(3.84, 0.64)$
    - $x_3 = vec(1.92, 0.08) - 0.1 vec(3.84, 0.64) = vec(1.536, 0.016)$

  Convergenza verso il minimo $(0, 0)$ dove $f(0,0) = 0$.
]

#attenzione()[
  *Scelta del Learning Rate $alpha$*:

  - $alpha$ troppo *piccolo*: convergenza molto lenta, molte iterazioni
  - $alpha$ troppo *grande*: oscillazioni, possibile divergenza
  - $alpha$ *ottimale*: bilanciamento tra velocità e stabilità

  In pratica si usano tecniche di *learning rate scheduling* (decay, adaptive methods come Adam).
]

=== Visualizzazione del Processo di Ottimizzazione

#figure()[
  #cetz.canvas({
    import cetz.draw: *

    // Assi
    line((-3, 0), (3, 0), mark: (end: ">"))
    content((3.3, 0), $x$)
    line((0, -0.5), (0, 3.5), mark: (end: ">"))
    content((0, 3.8), $f(x)$)

    // Funzione quadratica
    let points = ()
    for i in range(-30, 31) {
      let x = i / 10.0
      let y = x * x * 0.3 + 0.2
      points.push((x, y))
    }
    line(..points, stroke: (paint: blue, thickness: 2pt))

    // Traiettoria gradient descent
    let trajectory = (
      (2.2, 1.65),
      (1.5, 0.875),
      (0.9, 0.443),
      (0.5, 0.275),
      (0.2, 0.212),
      (0.05, 0.2075),
    )

    // Disegna traiettoria
    for i in range(trajectory.len() - 1) {
      let pt1 = trajectory.at(i)
      let pt2 = trajectory.at(i + 1)
      line(pt1, pt2, mark: (end: ">"), stroke: (paint: red, thickness: 1.5pt))
    }

    // Punti sulla traiettoria
    for (i, pt) in trajectory.enumerate() {
      circle(pt, radius: 0.08, fill: red, stroke: none)
      if i == 0 {
        content((pt.at(0), pt.at(1) + 0.3), text(size: 8pt, $x_0$), fill: red)
      } else if i == trajectory.len() - 1 {
        content((pt.at(0) - 0.3, pt.at(1)), text(size: 8pt, $x^*$), fill: red)
      }
    }

    // Minimo
    circle((0, 0.2), radius: 0.1, fill: green, stroke: (paint: green, thickness: 2pt))

    // Etichette
    content((1.5, 3), text(size: 10pt, $f(x) = x^2$), fill: blue)
    content((2, 2.3), text(size: 9pt, [Gradient Descent]), fill: red)
  })

  Visualizzazione del gradient descent su una funzione unidimensionale. La traiettoria rossa mostra i passi iterativi che convergono al minimo (punto verde).
]

== Composizione di Funzioni e Chain Rule

Nel deep learning, i modelli sono costruiti attraverso la *composizione di funzioni*. Ogni layer della rete applica una trasformazione, e l'output di un layer diventa l'input del successivo. Per calcolare i gradienti in questi modelli complessi, è fondamentale la *regola della catena* (chain rule).

=== La Regola della Catena (Chain Rule)

La *chain rule* è una regola fondamentale del calcolo differenziale che permette di calcolare la derivata di funzioni composte.

==== Caso unidimensionale
Date due funzioni $g: R -> R$ e $f: R -> R$ (derivabili), la derivata della composizione $h(x) = f(g(x))$ è:
$
  (d h)/(d x) = (d f)/(d g) dot (d g)/(d x)
$

oppure, in notazione più esplicita:
$
  h'(x) = f'(g(x)) dot g'(x)
$

#esempio()[
  Consideriamo $h(x) = (x^2 + 1)^3$.

  Possiamo vedere questa funzione come composizione:
  - $g(x) = x^2 + 1$
  - $f(g) = g^3$
  - $h(x) = f(g(x))$

  Applicando la chain rule:
  $
    h'(x) & = f'(g(x)) dot g'(x) \
          & = 3g^2 dot 2x \
          & = 3(x^2 + 1)^2 dot 2x \
          & = 6x(x^2 + 1)^2
  $
]

=== Caso multivariato

Siano $f: R^n -> R^m$ e $g: R^m -> R^k$ due funzioni differenziabili. Allora per ogni punto $x_0 in R^n$, lo *jacobiano* della funzione composta $h(x) = f(g(x))$ ($R^n -> R^k$) è data dal prodotto delle matrici Jacobiane:
$
  underbrace(J_h (x_0),k times n) = underbrace(J_g (f(x)), k times m) dot underbrace(J_f (x), m times n)
$

dove:
- $J_h (x) in R^(k times n)$: Jacobiano di $h$
- $J_g (f(x)) in R^(k times m)$: Jacobiano di $f$ valutato in $g(x)$
- $J_f (x) in R^(m times n)$: Jacobiano di $g$ valutato in $x$

#nota()[
  *Interpretazione per il Deep Learning*:

  In una rete neurale con $L$ layer, l'output finale è una composizione di funzioni:
  $
    y = f_L (f_(L-1)(dots f_2(f_1(x))))
  $

  Per calcolare il gradiente della loss rispetto ai parametri del primo layer, dobbiamo applicare ripetutamente la chain rule, moltiplicando i gradienti di tutti i layer attraversati.

  Questo è il principio alla base della *backpropagation*
]

#esempio()[
  Consideriamo una rete neurale minima con due layer (input e output scalari, un solo neurone nascosto, dimensone $1$):

  - Layer 1: $z = w_1 x + b_1$ (trasformazione lineare)
  - Activation: $a = sigma(z)$ (funzione sigmoide)
  - Layer 2: $hat(y) = w_2 a + b_2$ (output)
  - Loss: $L = 1/2 (hat(y) - y)^2$ (errore quadratico)

  Per calcolare $(partial L)/(partial w_1)$ (gradiente rispetto al peso del primo layer), applichiamo la chain rule:
  $
    (partial L)/(partial w_1) = underbrace((partial L)/(partial hat(y)),"Step"1) dot (partial hat(y))/(partial a) dot (partial a)/(partial z) dot underbrace((partial z)/(partial w_1),"Step"4)
  $

  dove:
  - $(partial L)/(partial hat(y)) = 2 dot 1/2 (hat(y)-y) =hat(y) - y$
  - $(partial hat(y))/(partial a) = w_2$ (in quanto sto derivando in $a$, rimane solo il coefficiente $w_2$)
  - $(partial a)/(partial z) = sigma(z)(1 - sigma(z))$ (derivata della sigmoide)
  - $(partial z)/(partial w_1) = x$ (sto derivando rispetto a $w_1$, quindi rimane solo il coefficiente $x$)

  Il gradiente finale è:
  $
    (partial L)/(partial w_1) = (hat(y) - y) dot w_2 dot sigma(z)(1 - sigma(z)) dot x
  $

  Calcolando il gradiente rispetto al peso $b_1$:
  $
    (partial L)/(partial b_1) = (partial L)/partial(hat(y)) dot partial(hat(y))/partial(a) dot partial(a)/partial(z) dot partial(z)/partial(b_1) \
    (hat(y)-y) dot w_2 dot sigma (z) (1 - sigma(z)) dot 1
  $
  Calcolando il gradiente rispetto al peso $w_2$:
  $
    partial(L)/partial(w_2) = partial(L)/partial(hat(y)) dot partial(hat(y))/partial(w_2)\

    (hat(y)-y) dot a

  $



]

#attenzione()[
  La chain rule è il meccanismo fondamentale che permette al *backpropagation* di propagare i gradienti all'indietro attraverso tutti i layer della rete, partendo dalla loss fino ai primi parametri.
]

== Lo Jacobiano: Gradiente Generalizzato

Quando lavoriamo con funzioni che mappano vettori in vettori (come i layer delle reti neurali), il gradiente non è sufficiente. Abbiamo bisogno dello *Jacobiano*. Si tratta dunque di una generalizzazione del gradiente per funzioni vettoriali.

Data una funzione *vettoriale* $F: R^n -> R^m$:
$
  F(overline(x)) = vec(f_1(overline(x)), f_2(overline(x)), dots.v, f_m(overline(x)))
$

dove $overline(x) = vec(x_1, x_2, dots, x_n) in R^n$, lo *Jacobiano* di $F$ è la matrice delle derivate parziali:
$
  J_F = mat(
    (partial f_1)/(partial x_1), (partial f_1)/(partial x_2), dots, (partial f_1)/(partial x_n);
    (partial f_2)/(partial x_1), (partial f_2)/(partial x_2), dots, (partial f_2)/(partial x_n);
    dots.v, dots.v, dots.down, dots.v;
    (partial f_m)/(partial x_1), (partial f_m)/(partial x_2), dots, (partial f_m)/(partial x_n)
  ) in R^(m times n)
$

Ogni *riga* $i$ contiene il gradiente della componente $f_i$:
$
  "Riga" i: nabla f_i (x) = vec((partial f_i)/(partial x_1), (partial f_i)/(partial x_2), dots, (partial f_i)/(partial x_n))
$

#nota()[
  *Confronto con il Gradiente*:

  - Il *gradiente* $nabla f$ si applica a funzioni scalari: $f: R^n -> R$ (produce un vettore $in R^n$)
  - Lo *Jacobiano* $J_F$ si applica a funzioni vettoriali: $F: R^n -> R^m$ (produce una matrice $in R^(m times n)$)

  In particolare, se $m=1$ (funzione scalare), lo Jacobiano coincide con il gradiente trasposto:
  $
    J_f = nabla f^T
  $
]

#esempio()[
  
  Consideriamo $F: R^2 -> R^2$ definita da:
  $
    F(x, y) = vec(f_1(x, y), f_2(x, y)) = vec(x^2 + y, x y)
  $

  *Calcoliamo lo Jacobiano*:

  Le componenti sono:
  - $f_1(x, y) = x^2 + y$
  - $f_2(x, y) = x y$

  Le derivate parziali sono:
  - $(partial f_1)/(partial x) = 2x$, $(partial f_1)/(partial y) = 1$
  - $(partial f_2)/(partial x) = y$, $(partial f_2)/(partial y) = x$

  Lo Jacobiano è una matrice $2 times 2$:
  $
    J_F (x, y) = mat(
      2x, 1;
      y, x
    )
  $

  In un punto specifico, ad esempio $(x, y) = (2, 3)$:
  $
    J_F (2, 3) = mat(
      4, 1;
      3, 2
    )
  $

]



#esempio()[
  *Jacobiano di un layer fully-connected*:

  Un layer lineare in una rete neurale è definito da:
  $
    z = W x + b
  $

  dove:
  - $x in R^n$: input
  - $W in R^(m times n)$: matrice dei pesi
  - $b in R^m$: bias
  - $z in R^m$: output

  Lo Jacobiano di questa trasformazione rispetto all'input $x$ è:
  $
    J_z (x) = (partial z)/(partial x) = W in R^(m times n)
  $

  Ogni elemento della matrice è:
  $
    (J_z)_(i j) = (partial z_i)/(partial x_j) = W_(i j)
  $

  Lo Jacobiano rispetto ai pesi $W$ è più complesso, ma segue lo stesso principio.
]

=== Jacobiano e Chain Rule

Lo Jacobiano è essenziale per applicare la chain rule in dimensione superiore. Supponiamo di avere una funzione $f: R^n -> R^m$ (layer che prende un input di dim $n$) e una funzione $g: R^m -> R$. Consideriamo la composizione $h(x) = g compose f$:
$
  h: R^n ->^f R^m ->^g R
$

Il gradiente di $h$ (rispetto all'input $x$, ovvero $h^'$) è dato dal prodotto dello Jacobiano di $f$ e del gradiente di $g$:
$
  h^' = gradient_x (g)= ((partial g)/(partial f_1) dot, dots, dot (partial g)/(partial f_m))^T\

  underbrace(J_f^T, n times m) dot underbrace(h^',m times 1) =  mat(
    (partial f_1)/(partial x_1), (partial f_2)/(partial x_1), dots, (partial f_m)/(partial x_1);
    (partial f_1)/(partial x_2), (partial f_2)/(partial x_2), dots, (partial f_m)/(partial x_2);
    dots.v, dots.v, dots.down, dots.v;
    (partial f_1)/(partial x_n), (partial f_2)/(partial x_n), dots, (partial f_m)/(partial x_n)
  ) dot vec((partial g)/(partial f_1), (partial g)/(partial f_2), dots, (partial g)/(partial f_m)) = vec(
    (partial g)/(partial x_1), (partial g)/(partial x_2), dots, (partial g)/(partial x_n)
  )
$

#informalmente()[
  - $f$ è l'output di un layer con $m$ neuroni (un vettore).
  
  - $g$ è la funzione di Loss finale che prende tutti questi output e calcola un numero unico (l'errore scalare $h$).

  - Il gradiente $gradient(g)$, è un vettore di dimensione $m$ che contiene l'errore per ciascuno degli $m$ neuroni di output.

  - Lo Jacobiano $J_f$ è una matrice che descrive come ogni input $x_j$ influenza ogni output $f_i$.

il vettore risultante $J_f^T dot gradient(g)$ ci dice come ogni input $x_j$ contribuisce all'errore totale, permettendoci di aggiornare i pesi del layer precedente in modo efficace durante la backpropagation.

In questo caso, la chain rule serve a  raccogliere tutti i pezzetti di errore che provengono da tanti neuroni successivi diversi ($m$) e sommarli correttamente per capire come aggiornare un singolo neurone precedente ($n$). Senza la matrice jacobiana, il neurone $x_1$ non saprebbe quale segnale di errore ascoltare, perché è collegato a mille cose diverse contemporaneamente.
]

Se si avessero ad esempio $3$ layer:
- Input $x$ entra nel Layer 1 $->$ produce $f$
- $f$ entra nel Layer 2 $->$ produce $g$
- $g$ entra nel Layer 3 $->$ produce $h$


La loss finale (errore) viene calcolata da $h$. Per sapere come aggiornare l'input iniziale $x$ (o i pesi del primo layer), applichiamo la *Chain Rule a cascata*. Invece di avere una sola matrice Jacobiana, moltiplichiamo tra loro le *matrici Jacobiane di ogni layer*:
$
  "Errore su" x = J_(f) dot J_(g) dot J_(h) dot underbrace(gradient(h), "vettore degli"\ "errori" in R^m)
$

Ogni matrice fa fare all'errore un _salto_ all'indietro di un layer.






#nota()[
  *Dimensioni nel prodotto*:
  $
    underbrace(R^(k times m), J_g) dot underbrace(R^(m times n), J_f) = underbrace(R^(k times n), J_h)
  $

  Le dimensioni intermedie $m$ si cancellano nel prodotto matriciale, come richiesto dalla composizione di funzioni.
]

#esempio()[
  Consideriamo una rete neurale con due layer lineari:

  - Layer 1: $z = W_1 x in R^m$ (con $x in R^n$)
  - Layer 2: $y = W_2 z in R^k$ (con $z in R^m$)

  Per calcolare $(partial y)/(partial x)$, usiamo la chain rule:
  $
    (partial y)/(partial x) = (partial y)/(partial z) dot (partial z)/(partial x) = W_2 dot W_1
  $

  Lo Jacobiano della composizione è il prodotto delle matrici dei pesi:
  $
    J_(y,x) = W_2 W_1 in R^(k times n)
  $
]

#attenzione()[
  *Importanza per la Backpropagation*:

  Durante la backpropagation, il gradiente della loss viene propagato all'indietro attraverso la rete moltiplicando ripetutamente gli Jacobiani dei vari layer. Questo è computazionalmente efficiente grazie alla struttura matriciale degli Jacobiani.
]

== Grafi Computazionali e Differenziazione Automatica

I framework moderni di deep learning (PyTorch, TensorFlow) implementano la *differenziazione automatica* (automatic differentiation, o *autograd*) attraverso *grafi computazionali*.

=== Cos'è un Grafo Computazionale

Un *grafo computazionale* è un grafo orientato aciclico (DAG - Directed Acyclic Graph) che rappresenta la sequenza di operazioni matematiche eseguite durante il calcolo di una funzione.

*Componenti del grafo*:
- *Nodi*: rappresentano *valori* (input, parametri, risultati intermedi, output)
- *Archi*: rappresentano *operazioni* matematiche ($+, -, times, div, "exp", "log"$, ecc.)

#nota()[
  Il grafo cattura la *struttura della funzione* e le *dipendenze* tra le variabili. Questo permette di calcolare automaticamente i gradienti usando la chain rule.
]

=== Come si Costruisce il Grafo

Il grafo viene costruito *dinamicamente* durante l'esecuzione del codice (*dynamic computational graph* in PyTorch) o *staticamente* prima dell'esecuzione (*static computational graph* in TensorFlow 1.x).

*Costruzione in PyTorch* (dinamica):

1. *Definizione dei tensori*: I tensori con `requires_grad=True` diventano *nodi foglia* del grafo
2. *Esecuzione delle operazioni*: Ogni operazione aritmetica crea un nuovo nodo nel grafo
3. *Registrazione delle dipendenze*: PyTorch registra quale operazione ha generato quale tensore
4. *Tracciamento automatico*: Il grafo viene costruito man mano che il codice viene eseguito

#esempio()[
  Consideriamo il seguente codice PyTorch:

  ```python
  import torch

  # Input e parametri
  x = torch.tensor([2.0], requires_grad=True)  # Nodo foglia
  w = torch.tensor([3.0], requires_grad=True)  # Nodo foglia
  b = torch.tensor([1.0], requires_grad=True)  # Nodo foglia

  # Forward pass: costruzione del grafo
  u = x * w       # Operazione di moltiplicazione
  z = u + b       # Operazione di addizione
  y = z ** 2      # Operazione di potenza
  ```

  PyTorch costruisce automaticamente il seguente grafo:

  #figure(
    cetz.canvas({
      import cetz.draw: *

      // Configurazione
      let node-style = (stroke: black, fill: blue.lighten(80%))
      let op-style = (stroke: black, fill: orange.lighten(70%))
      let param-style = (stroke: black, fill: green.lighten(70%))

      // Nodi input/parametri (foglie)
      circle((-2, 0), radius: 0.4, ..param-style, name: "x")
      content("x", text(size: 10pt, $x$))

      circle((0, 0), radius: 0.4, ..param-style, name: "w")
      content("w", text(size: 10pt, $w$))

      circle((2, 0), radius: 0.4, ..param-style, name: "b")
      content("b", text(size: 10pt, $b$))

      // Operazione moltiplicazione
      rect((-1, 1.5), (rel: (1, 0.6)), ..op-style, name: "mul")
      content((-0.5, 1.8), text(size: 9pt, $times$))

      // Nodo u
      circle((-0.5, 3), radius: 0.4, ..node-style, name: "u")
      content("u", text(size: 10pt, $u$))

      // Operazione addizione
      rect((0.5, 3.5), (rel: (1, 0.6)), ..op-style, name: "add")
      content((1, 3.8), text(size: 9pt, $+$))

      // Nodo z
      circle((1, 5), radius: 0.4, ..node-style, name: "z")
      content("z", text(size: 10pt, $z$))

      // Operazione potenza
      rect((0.5, 5.5), (rel: (1, 0.6)), ..op-style, name: "pow")
      content((1, 5.8), text(size: 9pt, $x^2$))

      // Nodo output y
      circle((1, 7), radius: 0.4, ..node-style, name: "y")
      content("y", text(size: 10pt, $y$))

      // Archi
      line("x", "mul", mark: (end: ">"), stroke: 1.5pt)
      line("w", "mul", mark: (end: ">"), stroke: 1.5pt)
      line("mul", "u", mark: (end: ">"), stroke: 1.5pt)
      line("u", "add", mark: (end: ">"), stroke: 1.5pt)
      line("b", "add", mark: (end: ">"), stroke: 1.5pt)
      line("add", "z", mark: (end: ">"), stroke: 1.5pt)
      line("z", "pow", mark: (end: ">"), stroke: 1.5pt)
      line("pow", "y", mark: (end: ">"), stroke: 1.5pt)

      // Legenda
      rect((3, 6), (rel: (2.5, 1.8)), stroke: 1pt)
      content((4.25, 7.5), text(size: 9pt, weight: "bold", [Legenda]))
      circle((3.3, 7), radius: 0.2, ..param-style)
      content((4, 7), text(size: 8pt, [Parametri/Input]), anchor: "west")
      circle((3.3, 6.5), radius: 0.2, ..node-style)
      content((4, 6.5), text(size: 8pt, [Nodi intermedi]), anchor: "west")
      rect((3.2, 6.1), (rel: (0.3, 0.2)), ..op-style)
      content((4, 6.2), text(size: 8pt, [Operazioni]), anchor: "west")
    }),
    caption: [Grafo computazionale per $y = (x times w + b)^2$. I nodi verdi sono parametri con `requires_grad=True`, i nodi blu sono valori intermedi, e i nodi arancioni sono operazioni.],
  )

  *Valori concreti*:
  Con $x=2$, $w=3$, $b=1$:
  $
    u & = 2 times 3 = 6 \
    z & = 6 + 1 = 7 \
    y & = 7^2 = 49
  $
]

=== Forward Pass e Backward Pass

Il grafo computazionale supporta due modalità di attraversamento:

*1. Forward Pass* (Propagazione in avanti):
- Si parte dai nodi *foglia* (input e parametri)
- Si attraversa il grafo seguendo la direzione degli archi
- Si calcolano i valori intermedi fino ad arrivare all'output
- *Risultato*: calcolo del valore della funzione

*2. Backward Pass* (Backpropagation):
- Si parte dal nodo *output* (loss)
- Si attraversa il grafo in direzione *opposta* agli archi
- Si applicano le regole di derivazione usando la chain rule
- *Risultato*: calcolo dei gradienti rispetto a tutti i parametri

#esempio()[
  Continuando l'esempio precedente, calcoliamo i gradienti con `y.backward()`:

  ```python
  # Backward pass
  y.backward()  # Calcola automaticamente i gradienti

  print(f"dy/dx = {x.grad}")  # Gradiente rispetto a x
  print(f"dy/dw = {w.grad}")  # Gradiente rispetto a w
  print(f"dy/db = {b.grad}")  # Gradiente rispetto a b
  ```

  *Applicazione manuale della chain rule*:

  $
    y & = z^2 quad       & => quad (d y)/(d z) & = 2z = 2(7) = 14 \
    z & = u + b quad     & => quad (d z)/(d u) & = 1, quad (d z)/(d b)     &     = 1 \
    u & = x times w quad & => quad (d u)/(d x) & = w = 3, quad (d u)/(d w) & = x = 2
  $

  *Gradienti finali* (applicando la chain rule):
  $
    (d y)/(d x) & = (d y)/(d z) dot (d z)/(d u) dot (d u)/(d x) = 14 dot 1 dot 3 = 42 \
    (d y)/(d w) & = (d y)/(d z) dot (d z)/(d u) dot (d u)/(d w) = 14 dot 1 dot 2 = 28 \
    (d y)/(d b) & = (d y)/(d z) dot (d z)/(d b) = 14 dot 1 = 14
  $

  PyTorch calcola automaticamente questi valori esplorando il grafo all'indietro!
]

=== Accumulazione dei Gradienti

Un aspetto importante della differenziazione automatica in PyTorch è l'*accumulazione dei gradienti*.

#attenzione()[
  Per default, PyTorch *accumula* i gradienti nel campo `.grad` di ogni tensore. Questo significa che chiamare `.backward()` multiple volte *somma* i nuovi gradienti a quelli esistenti.

  *Conseguenza pratica*: Nel training loop è necessario chiamare `optimizer.zero_grad()` o `tensor.grad.zero_()` prima di ogni backward pass per pulire i gradienti dell'iterazione precedente.
]

#esempio()[
  ```python
  x = torch.tensor([1.0], requires_grad=True)

  # Prima backward
  y1 = x ** 2
  y1.backward()
  print(x.grad)  # tensor([2.])  (dy1/dx = 2x = 2)

  # Seconda backward SENZA azzerare
  y2 = x ** 3
  y2.backward()
  print(x.grad)  # tensor([5.])  (2 + 3 = 5, accumula!)

  # Terza backward CON azzeramento
  x.grad.zero_()
  y3 = x ** 4
  y3.backward()
  print(x.grad)  # tensor([4.])  (corretto, gradiente pulito)
  ```
]

=== Vantaggi della Differenziazione Automatica

La differenziazione automatica attraverso grafi computazionali offre numerosi vantaggi:

*1. Precisione*: I gradienti sono calcolati in modo *esatto*, non approssimato (come avverrebbe con differenze finite)

*2. Efficienza*: La backpropagation calcola tutti i gradienti in un singolo attraversamento del grafo

*3. Flessibilità*: Supporta *qualsiasi* funzione differenziabile composta dalle operazioni primitive

*4. Automaticità*: Il programmatore non deve derivare manualmente le formule dei gradienti

#nota()[
  *Complessità computazionale*:

  - *Forward pass*: $O(n)$ dove $n$ è il numero di operazioni
  - *Backward pass*: $O(n)$ (stesso ordine di grandezza del forward!)

  Questo è un risultato notevole: calcolare *tutti* i gradienti costa circa quanto calcolare la funzione stessa.
]

== Cos'è un graidente

Partiamo da funzioni definite su uno spazio $D$ (input vettori) che mappano in un reale.

$gradient$ è un operatore che viene applicato a una funzioni $f$ (rappresnta una rete neurale):
$
  gradient f: R^D->R^D
$
#informalmente[
  un gradiente è un vettore di derivate parziali.
  Applichiamo la definizione di derivata solo che consideriamo il tasso di variazioni su una singola variabile, tenendo costanti le altri.
]
Si fa difierenziazione solo rispetto a una variabile ecc //aggiugnere formula

#esempio()[
  $
    f: R^2 -> R
  $
  Otteniamo un campo vettoriale:
  $
    delta f(x) = (alpha f) / (alpha x_1) hat(i)+ (alpha f) / (alpha x_2) hat(j)
  $
  Il grafico 3d ci dice che dato un vettore che parte in un punto ci dice il grado di pendenza di quella funzione. I vettori sono più o meno intesi (modulo del vettore). i vettori hanno direzioni pendenze e intesità diverse man mano che mi muovo nel dominio
]

Aggiungee latro esempio con due variabili.

=== Curve di livello

Sono importanti perchè quando usiamo il gradiente per raggiungere un punto di minimo in una funzione, usiamo le curve di livello. Un passo viene fatto da una cruva di livello ad un altra curva di livello (scegliendo una certa direzioni). Posso dstinguere le curve di livello in base a quelle che mi favoriscono la discesa verso il basso.

//aggiungere immagine campo vettoriale generato dal paraboloide

Il gradiente in ogni punto della funzione è perpendicolare alla tangente della curva di livello. Il gradiente viene interpretato come direzione

*Teo*: (direzione di massima discesa) //aggiungere
#informalmente()[
  Il segno serve altrimenti andre in salita
]
Lo sviluppo di taylor speiga il perchè:
$
  f(x) = f(x_0)+ gradient (f_x_0)^T dot (x-x_0)
$
#informalmente()[
  La funzione in un punto intorno da $x_0$. Se voglio usare la funzione di taylor per approssimare l'intorno di un punto devo ricorrere al gradiente moltiolicato (prodotto scalre)

  //riguardare
  Voglio valutare la funzione in x scritto come lo scostamento di x dal bho.\
  Vado a sostituire nella formula
  $
    x_i - alpha gradient(f_(x_o))
  $
]
Mi muovo verso un valore del dominio in cui la funzione ha valore minore. La migliore direzione è quella che coincide con $gradient f(X_0)$. Per il teorema se scelgo $d^*$ in questo modo ho una discesa ottimale in qualsiasi putno della funzione

=== algoritmo

//agiungere algoritmo
Il passo è $alpha_k$ (learning rate), potrebbe variari tutte le volte che ho un passo di iterazione tramite degli algoritmi

Criteri di stop. Vogliamo arrivare al minimo assoluto $gradient f(x_k) = 0$. Questa equazione è soddisfatta da un minimo relativo o minimi locali,oppure in caso di grafico a sella ho una variabile che mi fa salire e l'altra scendere.

Dobbiamo fissare una soglia $epsilon$. Se soo sotto la soglia mi fermo:

== Jacobiano
Quando abbiamo delle funzioni vettore->vettore sono delle funzioni non banali $R^n->R^m$ (layer dei modelli deep).

Lo jacobiano è la generalizzazione del gradiente per le matrici. Ogni componente della funzione $F(hat(x))$ è una funzione $f_x(overline(x)))$ a sua volta(composizoni di funzioni). Il gradiente viene esteso, deve differenziare rispetto alle funzioni e alle variabili:

La prima riga è la raccolta delle derivate parziali per la prima componente della funzione $F$. L'ultima riga è la derivata parziale rispetto all'ultima componente di $F$ grande.
#esempio()[
  Date $f_1 = x^2+y$ e $f_2 = x y$, ottengo $2x = (alpha f)/x$
]

#esempio()[
  Se prendo la funzione $h = g(overline(f))$. La f (composizoni di funzioni) è una variabile anche se deriva dal calcolo di $f_1(overline(x)), dots, f_k (overline(x))$.

  Se devo calcolare il gradiente di $h$ devo arrivare a moltiplicare lo jacobiao della funzione $f$ per la derivata priam di $g$:

  #esempio()[
    h è la composione di g e di f, posso dire che $h$ diepende da x, $h(overline(x))=g(f(x)))$.

    Posso usare la composizione in qualche modo //riguardare
  ]
]
=== regola della catena

Se predo due funzioni f e g e la loro composizione è derivabile ottendo $f'(g(x)) = r^k$

Posso fare la compsizione in quanto :$f(x_0)$ produce uan cosa di dim $m$ mentre $g$ prende $m$ e restituisce $k$:
$
  g(f(x_o)) -> z in R^k
$

#attenzione()[
  per calcolare i gradienti bisogna fare prodotti basati sugli jacobiani
]

== Loss

Una Loss :
$
  R^D -> R
$
è una funzione che mappa da $R^D$ a $R$ è l'ultimo step del procedimento visto fino adesso. L'obbiettivo è minimizzare questo funzione considerando il set di parameti $w$ come variabile.

Apllichiamo l'algoritmo di discesa dei gradienti per minimizzare la funzione

== Differenziazione automatica

Implementano il calcolo *esatto* (non approssimato) delle derivate di una funzione, si tratta di un meccanisomo di calcolo dei gradienti.

Vogliamo calcolare il gradiente della funzione loss avendo a disposizione un numero di elementi che mappano l'input sull'output (sono un numero elevato).

Potrei prendere ogni singolo elemento e farlo transistare nel mdoello -> calcolare la predizione -> caloclare l'errore -> modificare. Si tratta dell'approccio ideale. Tuttavia è meno robusto della discesa del gradiente stocastico e computazionalmente costoso

=== Graident stocastic

Scelgo a caso un elemento del dominio e sulal base dell'errore che si comette $L$, modifico i pesi. Un altro approccio più funzionante è basato su batch.

Lavora nel seguente modo:
$
  w_t+1 = w_t - n 1 / |B_t| sum_(n in B_t) gradient_w underbrace(l(f(x_n;w_t),y_n), "valore commesso medio")
$
Prendo un batch $B$ di tot elementi (piccola rispetto al dataset) e applico la variazione dei pesi sul valor medio di tutti i batch (non sul singolo esempalre)

Quello che succedo è che se prendo il gradiente medio calcolato sul batch è che non riesco ad avere la discesa ottimale in quel punto, tuttavia ho dei vantaggi :
- Molta velocità, meno computazioni
- Capacità di fare esapece dei minimi locali (in quanto è più rumoroso)

== Grafo computazionale

PyTorch implementa dietro le quinte autograd. Costuisce un grafo orientato (grafo diretto aicilico) che traccia tutte le operaizoni in tempo reale:
- I nodi sono le operaizoni ($+, -, *, "/"$)
- Gli archi sono l'interdipendenza tra questi nodi

Le aprime chiavi `requires_grad=True` associata ai tensori e l'operazione `backward()` ci permettono di fare il forward e modificare i pesi.

I gradienti vengono accumolati, ci sono un pool di gradienti che vengono utilizzati per modificare poi i pesi, vengono accumolati nel campo `.grad` di un tensore. La regola di Differenziazione viene applicata al grafo.

#esempio()[
  Regressione logistica (regressione per la classificazione).
  $D={x_i,y_i}$

  Se mi fermo al modello lineare ottengo un iperpiano, commette errori. Se prendo la funzione sigmoide o logistica ci permette di dire che se prendo una soglia (ad esempio $0.5$) ci permette di dire che se la rete emette $z$ (logits) da +/-inifnito la funzione di attivazione li riposta in un dominio più semplice, posso discriminarli in due classi 0 e 1.

  La funione di loos è la cross-entropy.

  Con $y=1$ (label True) mi trovo ad avere $-y log_2 sigma(z)$ se $y=0$ ho la funzinzone $1-log_2 sigma(z)$.

  Grado delle computazioni, prendo in input la variabile $x$ e $w$ (nel caso del forward la $x$ è variabile e $w$ costante, viceversa nella differenziazione). Esse danno forma alla variabile $z$ (ad esempio prodotto). La catenza forma un grafo ad alto livello. L'ultima parte è quella di decisione, il modello esce con dei valori

  //agiungere immagini
  A partre da $x$ e $w$ viene generata $u = x times w$. A questo punto lo sommo a $b$ (realizzo il livello lineare):
  $
    z = u + b
  $
  (Composizione di funzioni). In uscita avrò $hat(y) = theta(z)$ e infine calcolo la loss.

  Una volta ottenuto il valore di loos metto in atto la discesa del gradiente con la chain rule. La chain rule è data dalla derivata di tutte le variabili in gioco:

  - $t = (alpha l) / (alpha hat(y))$
  - $(alpha l) / a(alpha z)$
  - $(alpha z) / (alpha u)$
  - $(alpha u) / (alpha w)$

  la deriva di $(alpha L)/(alpha w)$ è data dal prodotto di tutte le derivate.
]

== SDG in PyTorch

`loss.backward()` fa tutto lui

`optimezer.step()` ci aggiunge una regola di aggiornamento.

```py
for epoch in range (n_epochs)
  for data, targat in dataloader:
    output = model(data)#costruisce un grafo
    loss = criteration(output, target)
    optimezer.zero_grad() #pulisce i buffer
    loss.backward()
    optimizer.step()#aggiorna i pesi
```

//aggoungere confronto mini-batch e singolo elemento
La disce del graideitne su un singolo elemnto alla volta è molto più smooth, tuttavia è molto costosa a livello computazionale.





