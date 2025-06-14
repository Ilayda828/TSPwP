#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <string.h>

#define MAX_CITIES 50000
#define K 20
#define START_LIMIT 100
#define MAX_REGION_CITIES 1000
#define GRID_SIZE 100

typedef struct {
    int id, x, y, visited;
    int region_id;
    double centrality_score;
    double connectivity_score;
    double penalty_efficiency;
} City;

typedef struct {
    int index;
    int distance;
} Neighbor;

typedef struct {
    int cities[MAX_REGION_CITIES];
    int count;
    int min_x, max_x, min_y, max_y;
    double center_x, center_y;
} Region;

City cities[MAX_CITIES];
Neighbor knn[MAX_CITIES][K];
int cityScores[MAX_CITIES];
Region regions[GRID_SIZE * GRID_SIZE];

int cityCount = 0;
int regionCount = 0;
int penalty = 0;

// ---------- Utility Functions ----------

// Euclidean mesafe, yuvarlanmış
int roundDistance(City a, City b) {
    long long dx = (long long)a.x - b.x;
    long long dy = (long long)a.y - b.y;
    return (int)round(sqrt((double)(dx*dx + dy*dy)));
}

// Girdi oku: önce penalty, sonra id x y üçlüleri
int readInput(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        printf("Error: Could not open file %s\n", filename);
        return 0;
    }
    if (fscanf(fp, "%d", &penalty) != 1) {
        printf("Error: Could not read penalty\n");
        fclose(fp);
        return 0;
    }
    cityCount = 0;
    while (fscanf(fp, "%d %d %d",
                  &cities[cityCount].id,
                  &cities[cityCount].x,
                  &cities[cityCount].y) == 3) {
        cities[cityCount].visited = 0;
        cities[cityCount].region_id = 0;
        cityCount++;
        if (cityCount >= MAX_CITIES) {
            printf("Warning: MAX_CITIES exceeded (%d)\n", MAX_CITIES);
            break;
        }
    }
    fclose(fp);
    return 1;
}

// Çıktıyı yaz: maliyet, şehir sayısı, sonra şehir id'leri
int writeOutput(const char *filename, int *tour, int size, long long cost) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        printf("Error: Could not open output file %s\n", filename);
        return 0;
    }
    fprintf(fp, "%lld %d\n", cost, size);
    for (int i = 0; i < size; i++) {
        fprintf(fp, "%d\n", cities[tour[i]].id);
    }
    fclose(fp);
    return 1;
}

// ---------- Region + KNN ----------

void createRegions() {
    // dünya sınırlarını bul
    int min_x = cities[0].x, max_x = cities[0].x;
    int min_y = cities[0].y, max_y = cities[0].y;
    for (int i = 1; i < cityCount; i++) {
        if (cities[i].x < min_x) min_x = cities[i].x;
        if (cities[i].x > max_x) max_x = cities[i].x;
        if (cities[i].y < min_y) min_y = cities[i].y;
        if (cities[i].y > max_y) max_y = cities[i].y;
    }
    // dinamik grid boyutları
    int grid_cols = (int)sqrt((double)cityCount / 500.0) + 1;
    int grid_rows = grid_cols;
    if (grid_cols > GRID_SIZE) grid_cols = GRID_SIZE;
    if (grid_rows > GRID_SIZE) grid_rows = GRID_SIZE;
    int cell_w = (max_x - min_x) / grid_cols + 1;
    int cell_h = (max_y - min_y) / grid_rows + 1;

    regionCount = 0;
    for (int r = 0; r < grid_rows; r++) {
        for (int c = 0; c < grid_cols; c++) {
            Region *R = &regions[regionCount];
            R->count = 0;
            R->min_x = min_x + c * cell_w;
            R->max_x = R->min_x + cell_w;
            R->min_y = min_y + r * cell_h;
            R->max_y = R->min_y + cell_h;
            R->center_x = (R->min_x + R->max_x) / 2.0;
            R->center_y = (R->min_y + R->max_y) / 2.0;
            regionCount++;
        }
    }
    // şehirleri bölgelere ata
    for (int i = 0; i < cityCount; i++) {
        int bestR = 0, bestD = INT_MAX;
        for (int r = 0; r < regionCount; r++) {
            Region *R = &regions[r];
            int dx = cities[i].x - R->center_x;
            int dy = cities[i].y - R->center_y;
            int d = abs(dx) + abs(dy);
            if (d < bestD && R->count < MAX_REGION_CITIES) {
                bestD = d;
                bestR = r;
            }
        }
        regions[bestR].cities[ regions[bestR].count++ ] = i;
        cities[i].region_id = bestR;
    }
}

int compareNeighbors(const void *a, const void *b) {
    return ((Neighbor*)a)->distance - ((Neighbor*)b)->distance;
}

void computeKNN() {
    for (int i = 0; i < cityCount; i++) {
        int cnt = regions[cities[i].region_id].count;
        Neighbor *buf = malloc(cnt * sizeof(Neighbor));
        int m = 0;
        // önce aynı bölgeden al
        for (int j = 0; j < cnt; j++) {
            int idx = regions[cities[i].region_id].cities[j];
            if (idx == i) continue;
            buf[m].index = idx;
            buf[m].distance = roundDistance(cities[i], cities[idx]);
            m++;
        }
        // bölgedeki komşuları kısalaştır
        qsort(buf, m, sizeof(Neighbor), compareNeighbors);
        int use = m < K ? m : K;
        for (int k = 0; k < use; k++) {
            knn[i][k] = buf[k];
        }
        for (int k = use; k < K; k++) {
            knn[i][k].index = -1;
            knn[i][k].distance = INT_MAX;
        }
        free(buf);
    }
}

// ---------- Başlangıç Şehirleri Seçimi ----------

void calculateCityScores() {
    // geometric center
    double cx = 0, cy = 0;
    for (int i = 0; i < cityCount; i++) {
        cx += cities[i].x;
        cy += cities[i].y;
    }
    cx /= cityCount;
    cy /= cityCount;
    double maxd = sqrt(cx*cx + cy*cy) * 2;
    if (maxd < 1) maxd = 1;
    for (int i = 0; i < cityCount; i++) {
        double dx = cities[i].x - cx, dy = cities[i].y - cy;
        double d = sqrt(dx*dx + dy*dy);
        cities[i].centrality_score = 1.0 - (d / maxd);
        int close = 0;
        for (int k = 0; k < K && knn[i][k].index != -1; k++) {
            if (knn[i][k].distance < penalty) close++;
        }
        cities[i].connectivity_score = (double)close / K;
        cities[i].penalty_efficiency = (close>0 ? (double)penalty / close : 0.0);
        double combined = 0.3*cities[i].centrality_score
                        + 0.4*cities[i].connectivity_score
                        + 0.3*cities[i].penalty_efficiency;
        int raw = (int)(combined * 10000);
        if (raw < 0) raw = 0;
        if (raw > 10000) raw = 10000;
        cityScores[i] = 10000 - raw;
    }
}

void selectDiverseStartCities(int *order, int maxStarts) {
    calculateCityScores();
    // puana göre sırala
    for (int i = 0; i < cityCount; i++) order[i] = i;
    for (int i = 0; i < cityCount - 1; i++) {
        for (int j = 0; j < cityCount - i - 1; j++) {
            if (cityScores[ order[j] ] > cityScores[ order[j+1] ]) {
                int t = order[j]; order[j] = order[j+1]; order[j+1] = t;
            }
        }
    }
    // çeşitlendirme
    int *sel = calloc(cityCount, sizeof(int));
    int *res = malloc(cityCount * sizeof(int));
    int cnt = 0;
    res[cnt++] = order[0];
    sel[ order[0] ] = 1;
    while (cnt < maxStarts && cnt < cityCount) {
        int best = -1, bestMin = -1;
        int lim = cityCount < maxStarts*3 ? cityCount : maxStarts*3;
        for (int i = 1; i < lim; i++) {
            int c = order[i];
            if (sel[c]) continue;
            int minD = INT_MAX;
            for (int j = 0; j < cnt; j++) {
                int o = res[j];
                int d = roundDistance(cities[c], cities[o]);
                if (d < minD) minD = d;
            }
            if (minD > bestMin) {
                bestMin = minD;
                best = c;
            }
        }
        if (best < 0) break;
        res[cnt++] = best;
        sel[best] = 1;
    }
    // kopyala
    for (int i = 0; i < cnt; i++) order[i] = res[i];
    free(sel);
    free(res);
}

// ---------- Greedy + 2-opt ----------

// Düzeltilmiş Deterministic Greedy tour - visited array'ini doğru kullanıyor
int greedyTour(int startCity, int *tour) {
    // CRITICAL: visited array'ini sıfırla
    for (int i = 0; i < cityCount; i++) {
        cities[i].visited = 0;
    }

    int cur = startCity;
    tour[0] = cur;
    cities[cur].visited = 1;
    int sz = 1;

    // Debug için başlangıç şehrini kontrol et
    if (startCity < 0 || startCity >= cityCount) {
        printf("Error: Invalid start city %d\n", startCity);
        return 1;
    }

    while (sz < cityCount) {
        int best = -1;
        int bestDistance = INT_MAX;

        // En yakın ziyaret edilmemiş şehri bul
        for (int i = 0; i < cityCount; i++) {
            if (!cities[i].visited) {
                int d = roundDistance(cities[cur], cities[i]);
                if (d < bestDistance) {
                    bestDistance = d;
                    best = i;
                }
            }
        }

        // Eğer ziyaret edilecek şehir kalmadıysa dur
        if (best == -1) {
            break;
        }

        // Debug: Aynı şehir tekrar ziyaret edilmemesini garanti et
        if (cities[best].visited) {
            printf("Error: Attempting to visit already visited city %d\n", best);
            break;
        }

        cur = best;
        cities[cur].visited = 1;
        tour[sz++] = cur;

        // Debug: İlerleme kontrolü (sadece çok büyük problemler için)
        if (cityCount > 30000 && sz % 5000 == 0) {
            printf("Greedy progress: %d/%d cities\n", sz, cityCount);
        }
    }

    return sz;
}

// Tur uzunluğu
long long calculateTourLength(int *tour, int size) {
    if (size < 2) return 0;

    long long sum = 0;
    for (int i = 0; i < size - 1; i++) {
        sum += roundDistance(cities[tour[i]], cities[tour[i+1]]);
    }
    sum += roundDistance(cities[tour[size-1]], cities[tour[0]]);
    return sum;
}

// 2-opt
void twoOpt(int *tour, int size) {
    if (size < 4) return;
    int improved = 1;
    int iterations = 0;
    const int maxIterations = (size > 10000) ? 50 : 1000; // Büyük problemler için iterasyon sınırı

    while (improved && iterations < maxIterations) {
        improved = 0;
        iterations++;
        int bestI = -1, bestJ = -1;
        int bestImprovement = 0;

        for (int i = 0; i < size-1; i++) {
            for (int j = i+2; j < size; j++) {
                if (j == size-1 && i == 0) continue;

                int A = tour[i], B = tour[i+1];
                int C = tour[j], D = tour[(j+1)%size];

                int oldD = roundDistance(cities[A], cities[B])
                         + roundDistance(cities[C], cities[D]);
                int newD = roundDistance(cities[A], cities[C])
                         + roundDistance(cities[B], cities[D]);

                int improvement = oldD - newD;
                if (improvement > bestImprovement) {
                    bestImprovement = improvement;
                    bestI = i;
                    bestJ = j;
                }
            }
        }

        if (bestImprovement > 0) {
            // En iyi swap'i uygula
            int a = bestI + 1, b = bestJ;
            while (a < b) {
                int t = tour[a]; tour[a] = tour[b]; tour[b] = t;
                a++; b--;
            }
            improved = 1;
        }
    }
}

// Basitleştirilmiş or-opt - sadece büyük problemler dışında kullan
void orOpt(int *tour, int size) {
    if (size < 4 || size > 15000) return; // Çok büyük problemlerde or-opt'u atla

    int improved = 1;
    int iterations = 0;
    const int maxIterations = 50;

    while (improved && iterations < maxIterations) {
        improved = 0;
        iterations++;

        // Sadece tek şehir taşıma (1-opt) - en güvenli versiyon
        for (int i = 0; i < size && !improved; i++) {
            int city = tour[i];
            int prev = (i - 1 + size) % size;
            int next = (i + 1) % size;

            // Bu şehri çıkarmanın maliyeti
            int removeCost = roundDistance(cities[tour[prev]], cities[city]) +
                           roundDistance(cities[city], cities[tour[next]]);
            int newEdgeCost = roundDistance(cities[tour[prev]], cities[tour[next]]);
            int removeGain = removeCost - newEdgeCost;

            // En iyi yerleştirme pozisyonunu bul
            int bestPos = -1;
            int bestGain = 0;

            for (int j = 0; j < size; j++) {
                // Mevcut pozisyon ve komşularını atla
                if (j == prev || j == i || j == next) continue;

                int nextJ = (j + 1) % size;

                // j ve j+1 arasına yerleştirmenin maliyeti
                int oldEdge = roundDistance(cities[tour[j]], cities[tour[nextJ]]);
                int newEdges = roundDistance(cities[tour[j]], cities[city]) +
                              roundDistance(cities[city], cities[tour[nextJ]]);
                int insertCost = newEdges - oldEdge;

                int totalGain = removeGain - insertCost;

                if (totalGain > bestGain) {
                    bestGain = totalGain;
                    bestPos = j;
                }
            }

            if (bestPos >= 0) {
                int tempCity = tour[i];

                if (i < bestPos) {
                    for (int k = i; k < bestPos; k++) {
                        tour[k] = tour[k + 1];
                    }
                    tour[bestPos] = tempCity;
                } else {
                    for (int k = i; k > bestPos + 1; k--) {
                        tour[k] = tour[k - 1];
                    }
                    tour[bestPos + 1] = tempCity;
                }

                improved = 1;
            }
        }
    }
}

long long solveWithNearestNeighbor(int *finalTour, int *finalSize) {
    printf("\n[3] Fast Heuristic (Nearest Neighbor + 2-opt + or-opt) begins...\n");
    clock_t start_time = clock();

    int sz = greedyTour(0, finalTour);
    if (sz != cityCount) {
        printf(" Warning: NN tour did not cover all cities (%d/%d)\n", sz, cityCount);
    }
    printf(" NN Tour created (%d cities). Optimization begins...\n", sz);

    twoOpt(finalTour, sz);
    orOpt(finalTour, sz);

    long long len = calculateTourLength(finalTour, sz);
    int missed = cityCount - sz;
    long long cost = len + (long long)missed * penalty;

    *finalSize = sz;

    clock_t end_time = clock();
    printf("Fast Heuristic completed in %.2f seconds\n",
           (double)(end_time - start_time) / CLOCKS_PER_SEC);
    printf("Fast Heuristic cost: %lld\n", cost);

    return cost;
}


int main(int argc, char *argv[]) {
    char inputFile[256]  = "input.txt";
    char outputFile[256] = "output.txt";
    if (argc >= 2) strncpy(inputFile,  argv[1], 255);
    if (argc >= 3) strncpy(outputFile, argv[2], 255);

    printf("Starting TSP Solver...\n");
    printf("Input file: %s\n",  inputFile);
    printf("Output file: %s\n", outputFile);

    clock_t T0 = clock();
    if (!readInput(inputFile)) return 1;
    printf("Read %d cities, penalty=%d\n", cityCount, penalty);

    int *startOrder = malloc(cityCount * sizeof(int));
    int *bestTour   = malloc(cityCount * sizeof(int));
    int *curTour    = malloc(cityCount * sizeof(int));
    if (!startOrder || !bestTour || !curTour) {
        printf("Error: Memory allocation failed\n");
        return 1;
    }

    long long bestCost = LLONG_MAX;
    int bestSize = 0;

    if (cityCount <= 2000) {
        printf("\nSmall problem detected (%d ≤ %d).\nUsing Region + KNN + Diverse Greedy + 2-opt\n",
               cityCount, 2000);

        createRegions();
        computeKNN();
        selectDiverseStartCities(startOrder, START_LIMIT);
        int tries = cityCount < START_LIMIT ? cityCount : START_LIMIT;

        for (int i = 0; i < tries; i++) {
            int s = startOrder[i];
            int sz = greedyTour(s, curTour);
            twoOpt(curTour, sz);
            orOpt(curTour, sz);
            long long len = calculateTourLength(curTour, sz);
            long long cost = len + (long long)(cityCount - sz) * penalty;
            if (cost < bestCost) {
                bestCost = cost;
                bestSize = sz;
                memcpy(bestTour, curTour, sz * sizeof(int));
            }
            if ((i+1) % 10 == 0 || i == tries-1) {
                printf(" Progress: %d/%d  Best cost: %lld\n",
                       i+1, tries, bestCost);
            }
        }

        printf("\nResults (Small problem pipeline):\n");
        printf("- Total cost: %lld\n", bestCost);
        printf("- Cities visited: %d\n", bestSize);

        if (!writeOutput(outputFile, bestTour, bestSize, bestCost)) {
            fprintf(stderr, "Error: Could not write output file %s\n", outputFile);
            return 1;
        }
        printf("Result written to %s\n", outputFile);
    }
    else if (cityCount <= 20000) {
        printf("\nMedium problem detected (%d ≤ %d).\nUsing NN + 2-opt heuristic\n",
               cityCount, 20000);

        int sz = greedyTour(0, curTour);
        twoOpt(curTour, sz);
        orOpt(curTour, sz);
        long long len   = calculateTourLength(curTour, sz);
        long long cost  = len + (long long)(cityCount - sz) * penalty;

        bestSize = sz;
        bestCost = cost;
        memcpy(bestTour, curTour, sz * sizeof(int));

        printf("\nResults (Medium problem pipeline):\n");
        printf("- Total cost: %lld\n", bestCost);
        printf("- Cities visited: %d\n", bestSize);

        if (!writeOutput(outputFile, bestTour, bestSize, bestCost)) {
            fprintf(stderr, "Error: Could not write output file %s\n", outputFile);
            return 1;
        }
        printf("Result written to %s\n", outputFile);
    }
    else {
        printf("\nExtreme problem detected (%d > %d). Using Nearest Neighbor only.\n",
               cityCount, 20000);

        int *tour = malloc(cityCount * sizeof(int));
        if (!tour) {
            fprintf(stderr, "Error: could not allocate tour array (%d cities)\n", cityCount);
            return 1;
        }

        int sz = greedyTour(0, tour);
        long long len    = calculateTourLength(tour, sz);
        int       missed = cityCount - sz;
        long long cost   = len + (long long)missed * penalty;


        if (!writeOutput(outputFile, tour, sz, cost)) {
            fprintf(stderr, "Could not write output file %s\n", outputFile);
            free(tour);
            return 1;
        }
        printf("Result written to %s\n", outputFile);

        free(tour);
        return 0;
    }


    free(startOrder);
    free(bestTour);
    free(curTour);

    clock_t T1 = clock();
    double elapsed = (double)(T1 - T0) / CLOCKS_PER_SEC;
    printf("Total execution time: %.2f seconds\n", elapsed);

    return 0;
}