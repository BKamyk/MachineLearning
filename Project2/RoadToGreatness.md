# ***Frogshop Clustering*** - How to See the Whole Without Seeing the Rest

The second task of the Machine Learning course was to analyze, cluster, and understand data about the Frogshop franchise.

This project is a prime example of why looking at data globally, rather than locally, might not always be beneficial.

---

## 1. ***First Dive***

The first step was to familiarize myself with the data. I started by displaying the headers of the available datasets to determine which ones deserved further attention.

The structure of the data can be summarized as follows:

**People order (*orders_dataset*) products (*products_dataset*) from sellers (*sellers_dataset*), who operate in various locations (*geolocation_dataset*).**

**They can also leave reviews and comments (*order_reviews_dataset*).**

---

## 2. ***Reviews*** $\leftrightarrow$ ***Comments***

As suggested by *the Taskmaster* (Dr. Turoboś), the first step was to explore the *order_reviews_dataset* and examine how reviews correlate with comments.

By plotting a countplot, I observed that most *review scores* are positive—about 57,000 have 5 stars. The second-largest group is 1-star reviews, numbering around 11,000.

I then split the reviews into two categories: those with comments and those without.  

The results were fascinating: *more people leave comments when they are unhappy.*

---

## 3. ***Products (Globally)***

The next suggestion from *the Taskmaster* was to analyze products and determine which ones were most highly regarded. 

The logical metric to use was **the reviews people made**, as they could classify the products.

*This is where the global approach failed for the first time.*

### Initial Step: Dimensionality Reduction

After removing null rows, I explored the dataset visually. Since most of the data was numerical, I decided to use dimensionality reduction methods to view it in 3D.

1. **PCA** (Principal Component Analysis): This revealed almost nothing.
2. **UMAP** (Uniform Manifold Approximation and Projection): This offered a beautiful structure that seemed worth analyzing. So I tried.

---

### $1^{st}$ Approach: *Analyzing the UMAP Structure*

The *umap_data* featured a distinct 'tail,' suggesting it might hold unique information. I isolated the data points from this tail, identified their corresponding products, and examined their reviews to see if they were linked to high- or low-quality items.

*Unfortunately, this tail exhibited the same density of reviews as the entire *products_dataset*.*

Although the 'tail' was not special in terms of user opinions, there was clearly **something** unique about the ***UMAP*** structure.

---

### $2^{nd}$ Approach: *Density of Reviews in the UMAP Structure*

Next, I investigated whether certain review types—e.g., 1-star reviews—showed any patterns within the ***UMAP*** structure.  

The result? **Nothing.**  

Every review type was distributed evenly, showing that the ***UMAP*** structure, while visually appealing, offered no useful insights.

---

### $3^{rd}$ Approach: *Exploring Different Labeling*

Finally, I tested whether other labels in the datasets might reveal meaningful patterns in the ***UMAP*** structure. I used product label names to create new categories, hoping to uncover non-trivial insights.

Again… **nothing.**

---

At this point, I should have recognized the issue: *There was no meaningful global structure.* Instead, I should have focused on analyzing specific product types or orders. But the allure of non-local features was too tempting.

---

## 4. ***Location of the Shops***

Next, I explored correlations between reviews and the geolocation of Frogshops.

The logical approach was to use customer reviews to identify the best shops across Brazil. By calculating the average rating for each shop, I hoped to locate the "top-performing" ones.

The result?  

**The best shops are everywhere.**

There was no correlation between shop location and average ratings. Once again, this conclusion stemmed from my attempt to uncover a global structure where none existed.

---

## 5. ***Conclusion***

This dataset was designed for local analysis, but my initial approach—using ***UMAP***—misled me into searching for global patterns.

The key takeaway? It’s essential to explore multiple angles and consider local features instead of relying on a single, overarching structure to explain everything.