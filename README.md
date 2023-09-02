# PDP with Mutations

This project is dedicated to the development of algorithms for solving the Pickup and Delivery Problem (PDP) using portfolio management of mutations.

## Method

In this project, the PDP is solved using evolutionary programming methods. Prior to optimization, a na&iuml;ve solution is constructed for a given set of orders. This solution represents a route that passes through all pickup points and then through all delivery points. The route constructed in this way does not violate the constraints of the problem regarding loading an order before its delivery, but it may not be the shortest route. After constructing the na¨ıve route, it is optimized using an evolutionary algorithm, where individuals represent different route variations.

The length of the route is used as a fitness function, and one of the available local changes is applied to the route as a mutation. Various configurations of the evolutionary algorithm were used during the experiments, each configuration existing in two variants: either parent individuals participate in selection or they do not.

- (1+1), (1,1) – Each generation has one individual producing one offspring.
- (1+N), (1,N) – Each generation has one individual producing N offspring.
- (K+KN), (K,KN) – Each generation has K individuals producing N offspring.

## Strategies and mutations

In previous research, the concept of a strategy was introduced, which is a set of rules combining mutations (mutations in this problem refer to various algorithms for making local changes to the route). A strategy is defined by a vector of probabilities representing the likelihood of choosing each mutation variant to be applied to the route. During the mutation stage of individuals in the evolutionary algorithm, the strategy is used to select a mutation. In previous research, five types of mutations were used: Lin-2-Opt, Double-Bridge, Point-Exchange, Couple-Exchange and Relocate-Block.

During the experiments, it was discovered that the Relocate-Block mutation outperformed the others in terms of the time required for its application to the route. This is because only the Relocate-Block mutation performs additional optimization by searching for the best position for a route segment throughout its length. However, the average route length reduction for the Relocate-Block mutation is significantly higher than for other mutations.

![Strategies convergence](../media/strategies.jpg?raw=true)
*The convergence of implemented strategies with Relocate-Block (left), without Relocate-Block (right).*

Therefore, a new mutation called Random-Relocate-Block was developed, which inserts the obtained route segment into a randomly chosen position. The representation of this mutation is shown in Fig. 1 Additionally, a Combined-2 mutation was introduced, which applies two basic mutations in sequence. A Combined-2 mutation consists of a pair of predetermined basic mutations, applying one after the other. There are 2N different instances of this mutation.

## Results and Insights

A graph reflecting the average reduction in route length achieved by each mutation and the average time taken by each mutation when applied to a route was constructed based on the statistics collected for all the mutations used.

![Mutations properties](../media/mutations.jpg?raw=true)
*The properties of all implemented mutations and mutation combinations &mdash; mean path improvement and execution time. Different colors indicate basic mutations used in mutation combinations.*

Colored clusters in the graph represent the six basic mutations from which different types of Combined-2 mutations were formed. It can be observed that all Combined-2 mutations form clusters based on the basic mutations they use. The cluster utilizing the Relocate-Block mutation, represented by a light green color, is the most useful but also the most costly in terms of application.

This clustering of mutations confirms the previously obtained results when comparing strategies. The Relocate-Block mutation is the most useful but also the most time-consuming mutation. The Random-Relocate-Block mutation proposed in this study retains a high degree of route reduction relative to other basic mutations, however works much faster than the Relocate-Block mutation. Basic mutations and their combinations as different Combined-2 mutations have distinct properties, allowing to choose a mutations set to use in strategies depending on the specific problem an its constraints based on the obtained statistics.

> Despite promising results, there are limitations to consider.
> The effectiveness of strategies and mutations may vary
> with different datasets and problem instances.
> Future research should test these approaches on diverse
> real-world datasets to determine their generalizability.

Thank you for your interest in my work!

## License

MIT
