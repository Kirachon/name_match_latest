#![allow(unused)]

#[cfg(feature = "new_engine")]
pub mod prelude {
    use super::*;
}
#[cfg(feature = "new_engine")]
pub mod db_pipeline;
#[cfg(feature = "new_engine")]
pub mod file_checkpointer;
#[cfg(feature = "new_engine")]
pub mod legacy_adapters;
#[cfg(feature = "new_engine")]
pub mod person_pipeline;

#[cfg(feature = "new_engine")]
pub trait Matcher {
    type ItemA;
    type ItemB;
    type MatchExplanation;
    // Return score 0-100 and explanation if matched; None if filtered out
    fn compare(&self, a: &Self::ItemA, b: &Self::ItemB) -> Option<(u32, Self::MatchExplanation)>;
}

#[cfg(feature = "new_engine")]
pub trait Partitioner<T> {
    // Return a partition key for blocking
    fn key(&self, item: &T) -> String;
}

#[cfg(feature = "new_engine")]
pub trait Checkpointer {
    // Persist a checkpoint token for a job and partition
    fn save(&mut self, job: &str, partition: &str, token: &str) -> anyhow::Result<()>;
    // Recover a checkpoint token if present
    fn load(&self, job: &str, partition: &str) -> anyhow::Result<Option<String>>;
}

#[cfg(feature = "new_engine")]
pub struct StreamEngine<M, PA, PB, Ck> {
    matcher: M,
    part_a: PA,
    part_b: PB,
    ck: Ck,
}

#[cfg(feature = "new_engine")]
impl<M, PA, PB, Ck> StreamEngine<M, PA, PB, Ck> {
    pub fn new(matcher: M, part_a: PA, part_b: PB, ck: Ck) -> Self {
        Self {
            matcher,
            part_a,
            part_b,
            ck,
        }
    }
}

#[cfg(feature = "new_engine")]
impl<M, PA, PB, Ck> StreamEngine<M, PA, PB, Ck>
where
    M: Matcher,
    PA: Partitioner<M::ItemA>,
    PB: Partitioner<M::ItemB>,
    Ck: Checkpointer,
{
    // Scaffolding only: interface for future pipeline
    pub fn stream_match<'a, I, J>(&mut self, _job: &str, _a: I, _b: J) -> anyhow::Result<()>
    where
        I: IntoIterator<Item = &'a M::ItemA>,
        J: IntoIterator<Item = &'a M::ItemB>,
        M::ItemA: 'a,
        M::ItemB: 'a,
    {
        // TODO: implement: partition by key, resume via checkpoint, batch, apply matcher, write/export
        Ok(())
    }
}

#[cfg(feature = "new_engine")]
pub struct NoopPartitioner;
#[cfg(feature = "new_engine")]
impl<T> Partitioner<T> for NoopPartitioner {
    fn key(&self, _item: &T) -> String {
        String::from("")
    }
}

#[cfg(feature = "new_engine")]
pub struct FnPartitioner<T, F: Fn(&T) -> String>(pub F, std::marker::PhantomData<T>);
#[cfg(feature = "new_engine")]
impl<T, F: Fn(&T) -> String> Partitioner<T> for FnPartitioner<T, F> {
    fn key(&self, item: &T) -> String {
        (self.0)(item)
    }
}

#[cfg(feature = "new_engine")]
impl<M, PA, PB, Ck> StreamEngine<M, PA, PB, Ck>
where
    M: Matcher,
    PA: Partitioner<M::ItemA>,
    PB: Partitioner<M::ItemB>,
    Ck: Checkpointer,
{
    pub fn for_each<'a, I, J, F>(
        &mut self,
        job: &str,
        a: I,
        b: J,
        mut on_match: F,
    ) -> anyhow::Result<usize>
    where
        I: IntoIterator<Item = &'a M::ItemA>,
        J: IntoIterator<Item = &'a M::ItemB>,
        M::ItemA: 'a,
        M::ItemB: 'a,
        F: FnMut(&M::ItemA, &M::ItemB, u32, &M::MatchExplanation) -> anyhow::Result<()>,
    {
        use std::collections::HashMap;
        let a_vec: Vec<&M::ItemA> = a.into_iter().collect();
        let b_vec: Vec<&M::ItemB> = b.into_iter().collect();
        // Group B by partition key
        let mut b_groups: HashMap<String, Vec<usize>> = HashMap::new();
        for (j, bj) in b_vec.iter().enumerate() {
            let k = self.part_b.key(bj);
            b_groups.entry(k).or_default().push(j);
        }
        // Iterate A by partition key
        let mut count = 0usize;
        let mut last_key = String::new();
        for (i, ai) in a_vec.iter().enumerate() {
            let k = self.part_a.key(ai);
            if k != last_key {
                let _ = self.ck.save(job, &k, &format!("i{}", i));
                last_key = k.clone();
            }
            if let Some(js) = b_groups.get(&k) {
                for &j in js {
                    if let Some((score, expl)) = self.matcher.compare(ai, b_vec[j]) {
                        on_match(ai, b_vec[j], score, &expl)?;
                        count += 1;
                    }
                }
            }
        }
        Ok(count)
    }
}
