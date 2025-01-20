Clippie: A little inference-only implementation of CLIP
=======================================================

Clippie is a simple, CPU-based, pure [Numpy](https://numpy.org/) implementation
of (the forward pass of) OpenAI's [CLIP](https://openai.com/research/clip)
image and text encoding deep neural network.


Usage
-----

Installation:

    $ pip install mossblaser-clippie

Get started like so:

    >>> from clippie import encode_text, encode_image
    
    >>> # Encode some text
    >>> text_vectors = encode_text([
    ...     "a toddler looking inside a computer",
    ...     "people walking along a mountain ridge",
    ...     "a beautiful lake",
    ... ])
    >>> text_vectors.shape  # (input_index, vector_dimension)
    (3, 512)
    
    >>> # Encode some images
    >>> from PIL import Image
    >>> image_vectors = encode_image([
    ...     Image.open("toddler.jpg"),
    ...     Image.open("mountain.jpg"),
    ...     Image.open("lake.jpg"),
    ... ])
    >>> image_vectors.shape  # (input_index, vector_dimension)
    (3, 512)
    
    >>> # Compute cosine similarity
    >>> import numpy as np
    >>> text_vectors /= np.linalg.norm(text_vectors, axis=1, keepdims=True)
    >>> image_vectors /= np.linalg.norm(image_vectors, axis=1, keepdims=True)
    >>> similarity = text_vectors @ image_vectors.T
    
    >>> # Note that the matching text/image pairs (on the diagonal) have the
    >>> # highest values as you would hope.
    >>> similarity
    array([[0.29675007, 0.0999563 , 0.12603459],
           [0.09451606, 0.25567788, 0.18573087],
           [0.1604508 , 0.17910984, 0.2590417 ]], dtype=float32)


Generating a Weights File
-------------------------

By default, Clippie will automatically download and use a copy of the CLIP
ViT-B-32 weights (preconverted into Clippie's format) from the corresponding
[GitHub release](https://github.com/mossblaser/clippie/releases/). (See
[weightie](https://github.com/mossblaser/weightie) for the weight storage
format and auto-download mechanism.).

You can also use the `clippie-convert-weights-file` script can be used to
convert a [PyTorch weights
file](https://github.com/openai/CLIP/blob/c5478aac7b9e007a2659d36b57ebe148849e542a/clip/clip.py#L36-L39)
from the reference CLIP implementation into the native weights
format:

    $ pip install path/to/clippie[convert]  # Extra packages needed for weights file conversion
    $ clippie-convert-weights-file /path/to/ViT-B-32.pt ViT-B-32.weights

The conversion script requires extra packages to be installed in order to
unpack the PyTorch weights file format (including PyTorch). After conversion,
these dependencies are no longer required.

You can then provide the path to the converted weights file (as a
[Path](https://docs.python.org/3/library/pathlib.html) object) to
`clippie.load` to load the needed weights and pass them to the encoder
functions:

    >>> from pathlib import Path
    >>> from clippie import load, encode_text, encode_image
    
    >>> # Load the weights...
    >>> weights = load(Path("/path/to/converted.weights"))
    
    >>> # Use them...
    >>> text_vectors = encode_text([...], weights.text_encoder)
    >>> image_vectors = encode_image([...], weights.image_encoder)

The converted weights file will typically be larger than the source CLIP
weights file because all values are expanded to float32 so that they can be
directly memory mapped by Clippie. Clippie is float32-only since most CPUs only
natively support down to float32 (and not float16).


Preemptive FAQ
--------------

**Why does Clippie exist?**

I wanted a decent search facility for my personal photo collection without
relying on a 3rd party service (e.g. Google Photos). Based on the impressive
results reported by
[various](https://mazzzystar.github.io/2022/12/29/Run-CLIP-on-iPhone-to-Search-Photos/)
other [projects](https://paulw.tokyo/post/real-time-semantic-search-demo/) it
became clear that OpenAI's CLIP model could work well, [in spite of photo
search explicitly out-of-scope for
CLIP](https://github.com/openai/CLIP/blob/main/model-card.md#out-of-scope-use-cases).
In fact, in my experience so far, search quality is substantially better than
Google Photos' search function.

To ensure I could build my photo search system on something which would remain
stable for some years, I wanted to avoid using anything based on a cutting-edge
deep learning framework -- ruling out the reference implementation and other
open source options. I am not in this for the research: I just want the tasty,
tasty search results!

Finally, I've been looking for a reason to finally learn more about deep
learning and this application was a good excuse. As you might expect from a
learning exercise, there is a perhaps slightly excessive quantity of commentary
in the code...


**Why not the [CLIP reference implementation](https://github.com/openai/CLIP)?**

By contrast with the reference implementation, Clippie has only a few
comparatively light-weight and stable dependencies (chiefly
[Numpy](https://numpy.org/) and
[Pillow](https://pillow.readthedocs.io/en/stable/)). As such, the largest
download needed is a copy of the weights, not gigabytes of software.
Furthermore, unlike most deep learning libraries -- which cater to a fast
moving field -- all of the dependencies used have been stable and well
supported for many years and are likely to remain so for many more years.

Separately, CLIP makes some slightly quirky choices in its implementation from
a software engineering point of view (e.g. its [quirky vocabulary binary
encoding format](./clippie/scripts/convert_vocab_file.py). As such, in several
places, Clippie does things slightly differently and, hopefully, a little more
clearly.


**Why CPU only?**

The smallest ViT-B/32 model can process an image or text string on my laptop's
CPU in 50-100 milliseconds and subjectively good quality results in searching
my collection of ~100k photos. This is plenty fast enough for (my) personal use
cases so no need to buy (or manage) a GPU!

Clippie's Numpy based implementation runs approximately as fast as the
PyTorch-based reference implementation on a CPU. Numpy appears to make fairly
effective use of available SIMD and multi-core facilities. That said, I'm
confident performance would be improved given a little profiling and effort.
For instance, no attention has been paid to memory layout or the effects of
batch sizes.


**Why float32 only?**

Whilst some of CLIP's Vision Transformer weights are given as 16 bit values,
with the exception of some recent ARM systems, most CPUs only support efficient
float32 arithmetic. This inflates memory usage somewhat but is faster on most
systems in practice.


**Why ViT-only?**

The CLIP authors reported that their [Vision Transformer
(ViT)](https://arxiv.org/abs/2010.11929)-based image encoder approach worked as
well or better than ResNet. Since the text encoder already uses a
[Transformer](https://arxiv.org/abs/1706.03762), I'd already done most of the
work implementing ViT and didn't fancy implementing the ResNet too.


**Why inference only?**

Since I'm only interested in *using* CLIP and the published weights work well I
had no need. That said, some of the limitations of OpenAI's training set
(presumably in the name of limiting potential abuse) do leave some gaps in
functionality. For example, the published weights are incapable of finding
pictures of breast feeding.

Separately, I'm especially keen to avoid falling down the rabbit hole of model
training lest I get sucked into deep learning research :).


**Does Clippie implement vector search?**

No.

Whilst various fancy [libraries](https://github.com/facebookresearch/faiss) and
[services](https://www.pinecone.io/) exist which implement (screamingly) fast
approximate nearest neighbour search on millions of vectors, they simply aren't
necessary at the scale of (my) personal photo collection. Naively using Numpy
as in the example code above can compute similarity of a search vector against
100k image vectors in about 50ms on my laptop.


**Does this reuse any code from the CLIP reference implementation?**

No -- though it does use its model weights and byte-pair encoding data.

This software is a from-scratch reimplementation of CLIP based almost entirely
on the descriptions in the original papers. However, to ensure
weight-compatibility, some parts of Clippie necessarily mimic the reference
implementation -- though no code has been reused or adapted.

Clippie does, however, re-use the data published alongside the reference CLIP
implementation:

* The vocabulary and byte-pair-encoding data included in the [(MIT
  Licensed)](https://github.com/openai/CLIP/blob/main/LICENSE) CLIP repository
  is also included in Clippie (albeit in a [different
  format](./clippie/scripts/convert_vocab_file.py)).
* The model weights provided with the reference CLIP implementation are also used
  (again, after [format conversion](#generating-a-weights-file)).
