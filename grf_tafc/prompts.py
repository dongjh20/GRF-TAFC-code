"""Prompt templates.

GENERATION_TEMPLATE  -> template T_G in Fig. 2 (formation generation)
REFLECTION_TEMPLATE  -> template T_R in Fig. 3 (execution-driven reflection)

The wording follows the complete examples released with the paper
(paper/generation-prompt-example.txt and paper/reflection-prompt-example.txt).
Fields in {braces} are filled by the grounded prompt composer.
"""

# Shared by both templates: system-derived task context (number of CAVs, lanes, previous formation).
TASK_CONTEXT_TEMPLATE = (
    "{cav_number} CAVs are traveling in formation on a {lane_number}-lane road, "
    "with a lane-wise distribution of {distribution} from Lane {first_lane} to Lane {last_lane}. "
    "The previous desired formation is: {previous_formation}. "
    "For each target position (x,y), x and y denote the longitudinal and lateral coordinates, "
    "respectively, and y indicates the lane index."
)

GENERATION_TEMPLATE = """Role and Task Specification

[Role]  You are an expert in multi-agent cooperative formation planning for connected and automated vehicles (CAVs).
[Task Description] Based on [Task Context] and [User Task Input], use the external resources specified in “Retrieval Instructions” and “Tool-Use Instructions” as needed, and follow “Reasoning Guidance” to generate an identity-free set of target positions satisfying [Task Requirements].
[Task Context] {task_context}
[User Task Input] {user_task_input}
[Task Requirements]
Road availability, traffic regulations, and formation-level safety conditions are hard constraints. Subject to these constraints:
1. Distribute the CAVs as evenly as possible across the selected usable lanes, such that the CAV counts in any two lanes differ by at most one.
2. Unless the user explicitly requests another geometry, arrange the CAVs in an interlaced pattern.
3. Prefer the candidate formation that requires the fewest lane changes during the transition from the previous formation, with each CAV permitted to move by at most one lane by default.
4. Unless otherwise required by the user task request or retrieved safety knowledge, preserve the previous same-lane longitudinal spacing of {spacing}.
5. Because only relative coordinates are required, the minimum x-coordinate in the generated formation is 0.
6. When sufficient lanes are available, prioritize translating the formation without changing its shape.

Retrieval Instructions

You have access to a structured knowledge database consisting of [Domain Knowledge] and [Experience Memory]. Use the provided context directly and retrieve only the additional knowledge needed for the current task. Among retrieved knowledge sources, give [Domain Knowledge] the highest priority.
[Domain Knowledge] The domain knowledge is stored as a graph covering human-intent interpretation, traffic regulations, and formation-level safety. Identify a task-relevant seed entity, invoke "explore_knowledge_graph", and follow relevant relations through multi-hop retrieval when necessary.
[Experience Memory] The experience memory consists of key-value text pairs recording previous task contexts and reflection-derived recommendations. Invoke "explore_experience" using the current task and operating context to retrieve the most relevant experiences.

Tool-Use Instructions

Invoke functions registered in "tool_list" when additional real-time information is required.

Reasoning Guidance

Follow the step-by-step reasoning workflow below:
1. Interpret [User Task Input] and determine the required formation adjustment.
2. Based on [Task Context], identify and acquire any missing knowledge or real-time information.
3. Determine the usable lanes, allocate the target positions across them, and generate relative coordinates satisfying [Task Requirements].
4. Verify the generated formation against the retrieved knowledge and real-time observations, particularly high-priority vehicles, obstacles, traffic incidents, and lane availability. If any requirement is not satisfied, return to Step 1 and regenerate the formation.

Output Format

Output the new desired formation within the tags: <formation>  </formation>. Sort the target positions first by y-coordinate in ascending order and then, for positions with the same y-coordinate, by x-coordinate in ascending order. Output the retrieved domain knowledge within the <knowledge> </knowledge> tags, the retrieved experience memory within the <experience> </experience> tags, the invoked tools within the <tool> </tool> tags, and the corresponding tool results within the <results> </results> tags. No other output is required.
"""

# Appended only when the output parser rejected the previous attempt (Sec. 3.2.3, regeneration).
VALIDATION_FEEDBACK_TEMPLATE = """
[Previous Output Validation] Your previous output was rejected by the output parser: {errors}. Regenerate a valid formation.
"""

REFLECTION_TEMPLATE = """Role and Task Specification

[Role] You are an expert in execution-outcome-based reflection for formation generation of connected and automated vehicles under varying task requirements.
[Task Description] Based on the reflection context including “What was requested”, “How the formation was generated”, and “What actually happened”,  follow “Reasoning Guidance” to generate the reflection insights specified in  “Output Format”.

What was requested

[Task Context] {task_context}
[User Task Input] {user_task_input}

How the formation was generated

[Retrieved Domain Knowledge] {retrieved_knowledge}
[Retrieved Experience Memory] {retrieved_experience}
[Invoked Real-Time Tools] {invoked_tools}
[Outputs of Real-Time Tools] {tool_outputs}
[Generated Formation] {generated_formation}

What actually happened

[Task Results] {task_results}
[Safety Checks] {safety_checks}
[Execution Performance] {execution_performance}

Reasoning Guidance

Follow the step-by-step reasoning workflow below:
1. Assess whether the generated formation satisfies the task requirements and whether it was successfully realized during execution.
2. Based on the user task input and task context, assess whether the retrieved domain knowledge and experience memory, invoked real-time tools, and their outputs were relevant and sufficient.
3. If the task was successfully completed, assess whether any suboptimal performance occurred. Proceed to Step 4 if suboptimal performance is identified; otherwise, proceed directly to Step 5. If the task was not successfully completed, proceed to Step 4.
4. Identify the primary cause of unsuccessful task completion or suboptimal performance, distinguishing among task interpretation, knowledge retrieval, tool use, formation generation, lower-level execution, and unexpected environmental changes.
5. Derive a concise and reusable reflection insight, including the applicable task context, recommended formation adjustment, and supporting rationale.

Output Format

Output the reflection insight within the <experience></experience> tags as a key-value pair, where the key specifies the applicable task context and the value specifies the recommended formation adjustment and supporting rationale. No other output is required.
"""
