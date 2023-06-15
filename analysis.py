from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain

from sqlmodel import Session, select
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

from database import SectionL, SectionChunk, Chunk, ChunkBatch, engine


sections = []
vectors = []
with Session(engine) as session:
    select_q = (
        select(
            SectionL,
            SectionChunk,
            ChunkBatch,
            Chunk,
        )
        .where(ChunkBatch.tag == "SmallSecFixes-Spacy-2000-05-26")
        .where(SectionL.chapter_id != 1)  # Hack to exclude table of contents
        .where(Chunk.batch_id == ChunkBatch.id)
        .where(SectionChunk.chunk_id == Chunk.id)
        .where(SectionL.id == SectionChunk.section_id)
        .order_by(SectionL.id)
    )
    results = session.exec(select_q)
    for r in results:
        sections.append(r[3])
        vectors.append(r[3].embedding)


# Choose the number of clusters, this can be adjusted based on the book's content.
# I played around and found ~10 was the best.
# Usually if you have 10 passages from a book you can tell what it's about
num_clusters = 12
kmeans = KMeans(
    n_clusters=num_clusters, n_init=100, max_iter=500, random_state=42, init="random"
).fit(vectors)

# for i in range(4490,4500):
#     dbscan = DBSCAN(eps=0.0001*i, min_samples=3, n_jobs=6).fit(vectors)
#     print(0.0001*i)
#     print(dbscan.labels_)

# Here are the clusters that were found. It's interesting to see the progression of clusters throughout the book. This is expected because as the plot changes you'd expect different clusters to emerge due to different semantic meaning
print(kmeans.labels_)


# Make list of the unique labels in order of their first appearance
unique_labels_ordered = []
for i in list(kmeans.labels_):
    if i not in unique_labels_ordered:
        unique_labels_ordered.append(i)

print(unique_labels_ordered)


matrix = np.array(vectors)
tsne = TSNE(
    n_components=2, perplexity=15, random_state=42, init="random", learning_rate=200
)
vis_dims = tsne.fit_transform(matrix)
# print(vis_dims.shape)


# Plot the reduced data
scatter = plt.scatter(vis_dims[:, 0], vis_dims[:, 1], c=kmeans.labels_, cmap="tab20")
# plt.colorbar(scatter)

# Create a custom legend with the desired colors
legend_elements = []
for color in sorted(list(set(kmeans.labels_))):
    legend_elements.append(
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=scatter.cmap(scatter.norm(color)),
            markersize=10,
        )
    )

# Plot the legend
plt.legend(
    legend_elements,
    sorted(list(set(kmeans.labels_))),
    loc="center left",
    bbox_to_anchor=(1, 0.5),
    ncol=1,
)

plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("Book Embeddings Clustered")
plt.show()


# Find the closest embeddings to the centroids

# Create an empty list that will hold your closest points
closest_indices = []
closest_ten_indices = []
ten_index_labels = []

# Loop through the number of clusters you have
for i in range(num_clusters):
    # Get the list of distances from that particular cluster center
    distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)

    # Find the list position of the closest one (using argmin to find the smallest distance)
    closest_index = np.argmin(distances)

    # Append that position to your closest indices list
    closest_indices.append(closest_index)

    # Do the same except 10
    ten_indices = list(np.argsort(distances)[:30])
    closest_ten_indices.append(ten_indices)

    index_labels = []
    for i in ten_indices:
        index_labels.append(kmeans.labels_[i])

    ten_index_labels.append(index_labels)

print(closest_indices)
selected_indices = sorted(closest_indices)
print(selected_indices)

book_order_indices = []
for label in unique_labels_ordered:
    book_order_indices.append(closest_indices[label])

print(book_order_indices)
print()


def print_nested(l):
    for x in l:
        print(x)
    print()


print_nested(closest_ten_indices)
print_nested(ten_index_labels)

# for i in book_order_indices:
#     print(sections[i].text.encode("utf-8"))
#     print()

# for i in unique_labels_ordered:
#     print(f"===\n{i}\n===")
#     for index in closest_ten_indices[i]:
#         print("```")
#         # print(sections[index].text.encode("utf-8"))
#         print(sections[index].text)
#         print("```")
#         print()
#     print()
