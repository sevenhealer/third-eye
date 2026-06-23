import { type Person, listIdentities } from './client'

// All enrolled persons, fetched lazily the first time a picker opens and
// cached until a new person is enrolled (mirrors vanilla's mergeAllPersons).
let allPersonsCache: Person[] | null = null

export function invalidatePersonsCache() {
  allPersonsCache = null
}

export async function getAllPersons(): Promise<Person[]> {
  if (allPersonsCache === null) {
    allPersonsCache = await listIdentities()
  }
  return allPersonsCache
}
