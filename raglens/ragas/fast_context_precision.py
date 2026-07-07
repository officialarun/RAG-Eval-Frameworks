import asyncio

from ragas.metrics._context_precision import ContextPrecision, QAC, Verification
from ragas.metrics.base import ensembler


class FastContextPrecision(ContextPrecision):
    """Evaluates all k contexts in parallel (asyncio.gather) instead of
    the default sequential loop. Drops job time from k×latency to 1×latency."""

    async def _ascore(self, row, callbacks):
        user_input, retrieved_contexts, reference = self._get_row_attributes(row)

        async def score_one(context):
            verdicts = await self.context_precision_prompt.generate_multiple(
                data=QAC(question=user_input, context=context, answer=reference),
                llm=self.llm,
                callbacks=callbacks,
            )
            return [v.model_dump() for v in verdicts]

        all_responses = await asyncio.gather(
            *[score_one(ctx) for ctx in retrieved_contexts]
        )

        answers = []
        for response in all_responses:
            agg = ensembler.from_discrete([response], "verdict")
            answers.append(Verification(**agg[0]))

        return self._calculate_average_precision(answers)
