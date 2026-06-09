// Third-Eye Neo4j Knowledge Graph — Schema and Seed Data

// ── Constraints (enforce uniqueness and create implicit indexes) ───────────────
CREATE CONSTRAINT person_id IF NOT EXISTS
    FOR (p:Person) REQUIRE p.id IS UNIQUE;

CREATE CONSTRAINT zone_id IF NOT EXISTS
    FOR (z:Zone) REQUIRE z.id IS UNIQUE;

CREATE CONSTRAINT camera_id IF NOT EXISTS
    FOR (c:Camera) REQUIRE c.id IS UNIQUE;

CREATE CONSTRAINT object_id IF NOT EXISTS
    FOR (o:Object) REQUIRE o.id IS UNIQUE;

// ── Indexes ────────────────────────────────────────────────────────────────────
CREATE INDEX person_name IF NOT EXISTS FOR (p:Person) ON (p.name);
CREATE INDEX zone_name IF NOT EXISTS FOR (z:Zone) ON (z.name);
CREATE INDEX object_class IF NOT EXISTS FOR (o:Object) ON (o.class);

// ── Seed Zones ─────────────────────────────────────────────────────────────────
MERGE (z1:Zone {id: 'building_entrance'})
  SET z1.name = 'Building Entrance', z1.type = 'entrance';

MERGE (z2:Zone {id: 'room_a'})
  SET z2.name = 'Room A', z2.type = 'general';

MERGE (z3:Zone {id: 'server_room'})
  SET z3.name = 'Server Room', z3.type = 'restricted';

MERGE (z4:Zone {id: 'corridor_main'})
  SET z4.name = 'Main Corridor', z4.type = 'monitored';

// Zone adjacency (for movement reasoning)
MERGE (z1)-[:ADJACENT_TO]->(z4);
MERGE (z4)-[:ADJACENT_TO]->(z2);
MERGE (z4)-[:ADJACENT_TO]->(z3);

// ── Seed Cameras ───────────────────────────────────────────────────────────────
MERGE (c1:Camera {id: 'cam-01'})
  SET c1.name = 'Entrance Camera', c1.status = 'active';
MERGE (c1)-[:COVERS]->(z1:Zone {id: 'building_entrance'});

MERGE (c2:Camera {id: 'cam-02'})
  SET c2.name = 'Server Room Camera', c2.status = 'active';
MERGE (c2)-[:COVERS]->(z3:Zone {id: 'server_room'});

// ── Seed Domain Objects ────────────────────────────────────────────────────────
MERGE (r1:Object {id: 'server-rack-1'})
  SET r1.name = 'Server Rack 1', r1.class = 'server_rack';
MERGE (r1)-[:LOCATED_IN]->(z3:Zone {id: 'server_room'});

MERGE (ai:Object {id: 'ai-cluster-1'})
  SET ai.name = 'AI Cluster', ai.class = 'compute_cluster';
MERGE (r1)-[:PART_OF]->(ai);

MERGE (d2:Object {id: 'desk-2'})
  SET d2.name = 'Desk 2', d2.class = 'desk';
MERGE (d2)-[:LOCATED_IN]->(z2:Zone {id: 'room_a'});

MERGE (ws1:Object {id: 'workstation-1'})
  SET ws1.name = 'Workstation', ws1.class = 'workstation';
MERGE (d2)-[:HAS]->(ws1);
