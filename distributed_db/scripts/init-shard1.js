try {
  rs.status();
  print("shard1RS already initialized");
} catch (e) {
  rs.initiate({
    _id: "shard1RS",
    members: [{ _id: 0, host: "shard1:27018" }],
  });
  print("shard1RS initialized");
}
