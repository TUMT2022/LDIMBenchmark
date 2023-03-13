cd ldimbenchmark
docker build -t ghcr.io/ldimbenchmark/ldimbenchmark:local .
cd ..
cd dualmethod
docker build --no-cache -t ghcr.io/ldimbenchmark/dualmethod:local .
cd ..
cd lila
docker build --no-cache -t ghcr.io/ldimbenchmark/lila:local .
cd ..
cd mnf
docker build --no-cache -t ghcr.io/ldimbenchmark/mnf:local .