library(igraph)
library(readr)

#--- load edge data ---
Reddit_side_efect_edgelist <- read_csv("/.../Reddit_side_efect_edgelist.csv") # replace "..." with your path to the file
PubMed_side_efect_edgelist <- read_csv("/.../PubMed_side_efect_edgelist.csv")
Twitter_side_efect_edgelist <- read_csv("/.../Twitter_side_efect_edgelist.csv")


names(Reddit_side_efect_edgelist) <- c("se_A", "se_B", "weight")
names(PubMed_side_efect_edgelist) <- c("se_A", "se_B", "weight")
names(Twitter_side_efect_edgelist) <- c("se_A", "se_B", "weight")


PubMed_side_efect_edgelist<-NULL

se <- rbind(Reddit_side_efect_edgelist,PubMed_side_efect_edgelist, Twitter_side_efect_edgelist)


# --- load stem dictionary ---
pmd_side_effect_stem_dictionary <- read_csv("/.../pmd_side_effect_stem_dictionary.csv")
Red_side_effect_stem_dictionary <- read_csv(".../Red_side_effect_stem_dictionary.csv")
TW_side_effect_stem_dictionary <- read_csv("/.../TW_side_effect_stem_dictionary_Twitter.csv")

pmd_side_effect_stem_dictionary<-NULL

side_effect_stem_dictionary <- rbind(pmd_side_effect_stem_dictionary, Red_side_effect_stem_dictionary, TW_side_effect_stem_dictionary)
side_effect_stem_dictionary <- side_effect_stem_dictionary[!duplicated(side_effect_stem_dictionary$stem),]


#--- clean edge data

library(sqldf)
se<-sqldf("select se_A, se_B, sum(weight) as weight from se group by se_A, se_B")

se = se[se$weight >= 3,] # at least 3 co-mentions


# remove non-side effect nodes
to_remove <- c('diabet',
               'obes',
               'injuri',
               'pain',
               'alcoholic',
               'alcohol intoxication',
               'alcohol',
               'red',
               'acn',
               'autism',
               'clumsiness',
               'death',
               'disabl',
               'diabetes',
               'diabetes burn',
               'diabetic coma',
               'diabetic depression',
               'diabetic ketoacidosis',
               'diabetic ketosis',
               'diabetic neuropathy',
               'diabetic ulcers',
               'dyslexia',
               'eczema',
               'fast',
               'hunger pains',
               'injury',
               'poison',
               'osteoporosi',
               'neuropathy pain',
               'numbing pain',
               'obesity',
               'painful cramping',
               'painful death',
               'parkinsons',
               'violent vomit',
               'vulvovaginal',
               'acne',
               'alcohol nausea',
               'aching pain',
               'cold sweats',
               'parkinson',
               'fibrosi',
               'append',
               'astma',
               'coma',
               'deaf',
               'sick',
               'infect',
               'suicid',
               'abort'
               )

se<-se[(!se$se_A %in% to_remove) & (!se$se_B %in% to_remove),]


#--- side effects of vann diagram ---
reddit <- union(Reddit_side_efect_edgelist$se_A, Reddit_side_efect_edgelist$se_B)
pmd <- union(PubMed_side_efect_edgelist$se_A, PubMed_side_efect_edgelist$se_B)
tw <- union(Twitter_side_efect_edgelist$se_A, Twitter_side_efect_edgelist$se_B)


reddit<-reddit[!reddit %in% to_remove]
pmd<-pmd[!pmd %in% to_remove]
tw<-tw[!tw %in% to_remove]

intersect_red_pmd <- intersect(reddit, pmd)
intersect_red_tw_pmd <- intersect(intersect_red_pmd, pmd)

#---- replace with real side effect name (not stem) ----

for(stem in intersect_red_tw_pmd){
  reddit[reddit==stem] = side_effect_stem_dictionary[side_effect_stem_dictionary$stem == stem, ]$entity
  tw[tw==stem] = side_effect_stem_dictionary[side_effect_stem_dictionary$stem == stem, ]$entity
  pmd[pmd==stem] = side_effect_stem_dictionary[side_effect_stem_dictionary$stem == stem, ]$entity
  intersect_red_tw_pmd[intersect_red_tw_pmd==stem] = side_effect_stem_dictionary[side_effect_stem_dictionary$stem == stem, ]$entity
}

se$entity_A <- 'NA'
for(stem in se$se_A){
  
  se[se$se_A==stem,'entity_A'] <- side_effect_stem_dictionary[side_effect_stem_dictionary$stem == stem, ]$entity
  
}



se$entity_B <- 'NA'
for(stem in se$se_B){
  
  se[se$se_B==stem,'entity_B'] <- side_effect_stem_dictionary[side_effect_stem_dictionary$stem == stem, ]$entity
  
}


# delete side effects

to_delete<- c(  'neoplasms',
                'thyroid carcinoma',
                'thyroid neoplasm',
                'periodontitis',
                'colitis',
                'overdose',
                'ga',
                'pregnancy',
                'mass',
                'psoriasis',
                'epilepsy',
                'tachyphylaxis',
                'cerebral',
                'epoptosis',
                'apoptosis',
                'burn injury',
                'atrophy',
                'ach',
                'hypothermia',
                'insulinomas',
                'obese asthma',
                'necrosis',
                'carcinomas',
                'cervical',
                'pancreatic',
                'pancreatic carcinoma',
                'fistula',
                'mastitis',
                'adhesions',
                'sepsis',
                'shock',
                'sweats',
                'obstruction',
                'wounding',
                'neutrophilia',
                'cerebral arteriosclerosis',
                'cerebral infarction',
                'asthma',
                'microangiopathy',
                'pancreatic dysplasia'
              )


se<-se[(!se$entity_A %in% to_delete) & (!se$entity_B %in% to_delete),]

reddit<-reddit[!reddit %in% to_remove]
tw <- reddit[!tw %in% to_remove]
pmd <- pmd[!pmd %in% to_remove]
intersect_red_tw_pmd<- intersect_red_tw_pmd[!intersect_red_tw_pmd %in% to_remove]

# Replace: make corrections to node names

se[se$entity_A == 'hepatic inflammation', 'entity_A'] <- 'hepatic'
se[se$entity_B == 'hepatic inflammation', 'entity_B'] <- 'hepatic'

intersect_red_tw_pmd[intersect_red_tw_pmd == "hepatic inflammation"] <- "hepatic"
reddit[reddit == "hepatic inflammation"] <- "hepatic"
tw[tw == "hepatic inflammation"] <- "hepatic"
pmd[pmd == "hepatic inflammation"] <- "hepatic"

se[se$entity_A == 'nephrotoxic nephritis', 'entity_A'] <- 'nephrotoxic'
se[se$entity_B == 'nephrotoxic nephritis', 'entity_B'] <- 'nephrotoxic'

intersect_red_tw_pmd[intersect_red_tw_pmd == 'nephrotoxic nephritis'] <- "nephrotoxic"
reddit[reddit == "nephrotoxic nephritis"] <- "nephrotoxic"
tw[tw == "nephrotoxic nephritis"] <- "nephrotoxic"
pmd[pmd == "nephrotoxic nephritis"] <- "nephrotoxic"

se[se$entity_A == 'pain hypersensitivity', 'entity_A'] <- 'hypersensitivity'
se[se$entity_B == 'pain hypersensitivity', 'entity_B'] <- 'hypersensitivity'
intersect_red_tw_pmd[intersect_red_tw_pmd == 'pain hypersensitivity'] <- "hypersensitivity"
reddit[reddit == "pain hypersensitivity"] <- "hypersensitivity"
tw[tw == "pain hypersensitivity"] <- "hypersensitivity"
pmd[pmd == "pain hypersensitivity"] <- "hypersensitivity"

se[se$entity_A == 'diarrhoea', 'entity_A'] <- 'diarrhea'
se[se$entity_B == 'diarrhoea', 'entity_B'] <- 'diarrhea'
intersect_red_tw_pmd[intersect_red_tw_pmd == 'diarrhoea'] <- "diarrhea"
reddit[reddit == "diarrhoea"] <- "diarrhea"
tw[tw == "diarrhoea"] <- "diarrhea"
pmd[pmd == "diarrhoea"] <- "diarrhea"


se$se_A<-NULL
se$se_B<-NULL
se<-se[,c('entity_A','entity_B','weight')]

#--- create graph --------------------------------------------------------------

g<-graph_from_data_frame(se, directed = FALSE)


# keep only the main component of a graph
components <- components(g)
g <- induced_subgraph(g, which(components$membership == which.max(components$csize)))

node_degrees = degree(g)

#--- clusters nodes ---
# Apply the Louvain community detection algorithm
communities <- cluster_louvain(g)
V(g)$cluster <- communities$membership

# Define colors for the clusters (you can customize this)
cluster_colors <- rainbow(max(communities$membership))

# Plot the graph with nodes colored by cluster
plot(g, 
     edge.arrow.size=.2, 
     vertex.color = cluster_colors[V(g)$cluster],
     vertex.label.color="black", 
     vertex.size=node_degrees,
     edge.width=E(g)$weight/50,
     vertex.label.cex = 0.7,
     layout = layout_on_grid
)


# save edgelist
edgelist <- data.frame(get.edgelist(g), E(g)$weight)
names(edgelist) <- c('node_from', 'node_to','weight')
write.csv(edgelist, row.names = FALSE,file='/.../g_edgelist.csv')

#--- centrality ----------------------------------------------------------------

nodes <- data.frame(node=V(g)$name)

nodes$degree <- degree(g)
nodes$bet<-betweenness(g)
nodes$closeness = closeness(g)
nodes$eigenvector = eigen_centrality(g)$vector
nodes$pagerank = page.rank(g)$vector
nodes$cc<-transitivity(g, type = "local")


# Get the membership vector indicating which community each node belongs to
nodes$community <- communities$membership

# write.csv(nodes[nodes$community==1,c('node')], row.names = FALSE)

write.csv(nodes, row.names = FALSE, file="/.../nodes.csv")


#--- statistics ---------------

res <- rbind(
                    data.frame(
                      'entity' = se$entity_A,
                      "weight" = se$weight
                    ),
                    data.frame(
                      'entity' = se$entity_B,
                      'weight'  = se$weight
                    )
)


res <- sqldf("select entity, sum(weight) as mention_freq from res group by entity")

library(readxl)
nodes <- read_excel("/.../nodes.xlsx")
nodes <- nodes[,c(1:3)]


#-- plot -- 
library(ggplot2)
library(viridis)
library(hrbrthemes)

# Example code with facet_wrap
ggplot(nodes, aes(x = `Side Effect`, y = Mention_Frequency)) +
  geom_bar(stat = "identity", fill = "blue") +
  labs(title = "Side Effects by Group Affiliation", x = "Side Effect", y = "Mention Frequency") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  facet_wrap(~`Group Affiliation`, scales = "free_x")


#--- overlap ----

intersect_red_tw_pmd<- intersect_red_tw_pmd[intersect_red_tw_pmd %in% nodes$`Side Effect`]

tw_non_overlap<- tw[tw %in%  nodes$`Side Effect`]
tw_non_overlap<- tw_non_overlap[!tw_non_overlap %in% intersect_red_tw_pmd ]
write.csv(tw_non_overlap, row.names = FALSE)

pmd_non_overlap<- pmd[pmd %in%  nodes$`Side Effect`]
pmd_non_overlap<- pmd_non_overlap[!pmd_non_overlap %in% intersect_red_tw_pmd ]
write.csv(pmd_non_overlap, row.names = FALSE)

reddit_non_overlap <- reddit[reddit %in%  nodes$`Side Effect`]
reddit_non_overlap<- reddit_non_overlap[!reddit_non_overlap %in% intersect_red_tw_pmd ]
write.csv(reddit_non_overlap, row.names = FALSE)

#--- Calculate probability of moving from one node to another -------------------

# Set the starting node and destination node
start_node <- "arthritis"
dest_node <- "anxiety"

# Get the neighbors of the starting node
neighbors <- list(neighbors(g, start_node, mode = 'all'))

# Calculate the probability of transitioning to the destination node
transition_probability <- 0

# Get the weighted edge list
weighted_edgelist <- data.frame(get.edgelist(g),E(g)$weight)
names(weighted_edgelist) <- c("from", "to", "weight")

if ( are.connected(g, start_node, dest_node)) {
  # Calculate the probability as the reciprocal of the number of neighbors
  w = weighted_edgelist[weighted_edgelist$from == start_node & 
                          weighted_edgelist$to == dest_node,]$weight
  transition_probability <-  w/as.vector(strength(g)[start_node])
}

cat("Transition probability from", start_node, "to", dest_node, "is:", transition_probability, "\n")




